// Forward-only counterpart of SkyJepaRotorPlant::step. One CUDA thread per
// candidate, with motor lag, gyroscopic torque, body-axis drag and SO(3).
__device__ void normalize3(float* v, int fallback) {
    float n = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    for (int i=0; i<3; ++i) v[i] = n > 1e-6f ? v[i]/n : (i == fallback ? 1.0f : 0.0f);
}
extern "C" __global__ void skyjepa_nominal_f32(
    const float* initial, const float* actions, const float* initial_motors, float* output,
    unsigned int batch, unsigned int steps, float dt, unsigned int substeps,
    float mass, float ix, float iy, float iz, float tau, float dx, float dy, float dz,
    float thrust_scale, float torque_scale, float arm, float gravity, float max_tw) {
    unsigned int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= batch) return;
    float x[18], motors[4];
    for (int i=0; i<18; ++i) x[i] = initial[(unsigned long long)candidate*18+i];
    for (int i=0; i<4; ++i) motors[i] = initial_motors[(unsigned long long)candidate*4+i];
    float inertia[3] = {ix,iy,iz}, drag[3] = {dx,dy,dz};
    float h = dt/substeps, response = 1.0f-expf(-h/tau), maximum = mass*gravity*max_tw/4.0f;
    for (unsigned int t=0; t<steps; ++t) {
        const float* action = actions+((unsigned long long)candidate*steps+t)*4;
        for (unsigned int sub=0; sub<substeps; ++sub) {
            float force[4], total=0.0f;
            for (int i=0; i<4; ++i) {
                motors[i] += (fminf(maximum,fmaxf(0.0f,action[i]))-motors[i])*response;
                force[i] = motors[i]*thrust_scale;
                total += force[i];
            }
            float torque[3] = {arm*(force[1]-force[3]), arm*(force[2]-force[0]),
                0.025f*torque_scale*(force[0]-force[1]+force[2]-force[3])};
            float iw[3] = {ix*x[15],iy*x[16],iz*x[17]};
            float gyro[3] = {x[16]*iw[2]-x[17]*iw[1], x[17]*iw[0]-x[15]*iw[2], x[15]*iw[1]-x[16]*iw[0]};
            for (int i=0; i<3; ++i) x[15+i] += ((torque[i]-gyro[i]-0.015f*x[15+i])/inertia[i])*h;
            float a=x[15]*h, b=x[16]*h, c=x[17]*h, theta=sqrtf(a*a+b*b+c*c), dr[9];
            if (theta < 1e-6f) {
                float small[9]={1,-c,b,c,1,-a,-b,a,1};
                for(int i=0;i<9;++i) dr[i]=small[i];
            } else {
                a/=theta; b/=theta; c/=theta;
                float co=cosf(theta), si=sinf(theta), q=1.0f-co;
                float rotation[9]={co+a*a*q,a*b*q-c*si,a*c*q+b*si,
                    b*a*q+c*si,co+b*b*q,b*c*q-a*si,c*a*q-b*si,c*b*q+a*si,co+c*c*q};
                for(int i=0;i<9;++i) dr[i]=rotation[i];
            }
            float r[9];
            for(int row=0;row<3;++row) for(int col=0;col<3;++col)
                r[row*3+col]=x[6+row*3]*dr[col]+x[7+row*3]*dr[3+col]+x[8+row*3]*dr[6+col];
            float cx[3]={r[0],r[3],r[6]}, cy[3]={r[1],r[4],r[7]};
            normalize3(cx,0);
            float dot=cx[0]*cy[0]+cx[1]*cy[1]+cx[2]*cy[2];
            for(int i=0;i<3;++i) cy[i]-=dot*cx[i];
            normalize3(cy,1);
            float cz[3]={cx[1]*cy[2]-cx[2]*cy[1],cx[2]*cy[0]-cx[0]*cy[2],cx[0]*cy[1]-cx[1]*cy[0]};
            for(int i=0;i<3;++i) { x[6+i*3]=cx[i]; x[7+i*3]=cy[i]; x[8+i*3]=cz[i]; }
            float body_drag[3];
            for(int i=0;i<3;++i) body_drag[i]=(x[6+i]*x[3]+x[9+i]*x[4]+x[12+i]*x[5])*drag[i];
            for(int i=0;i<3;++i) {
                float dw=x[6+i*3]*body_drag[0]+x[7+i*3]*body_drag[1]+x[8+i*3]*body_drag[2];
                float accel=x[8+i*3]*total/mass-(i==2 ? gravity : 0.0f)-dw/mass;
                x[3+i]+=accel*h; x[i]+=x[3+i]*h;
            }
            if(x[2]<0.05f) { x[2]=0.05f; x[5]=fmaxf(x[5],0.0f); }
        }
        for(int i=0;i<18;++i) output[((unsigned long long)candidate*steps+t)*18+i]=x[i];
    }
}
