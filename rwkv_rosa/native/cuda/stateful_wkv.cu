// Inference-only stateful counterpart of the vendored @rwkv3 forward.
// Same BF16 inputs, FP32 accumulator, decay, update and reduction order.
// State layout [B,H,value,key]. Input state is NEVER modified in place.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
using bf = __nv_bfloat16;
constexpr int N=64, CHUNK=16;
__global__ void stateful_forward(int T,int H,const bf* r_,const bf* w_,const bf* k_,
    const bf* v_,const bf* a_,const bf* b_,const float* initial,bf* out,float* final,int state_tokens) {
    int batch=blockIdx.y,head=blockIdx.x,i=threadIdx.x;
    long long offset=((long long)batch*H+head)*N*N;
    float state[N];
    #pragma unroll
    for(int j=0;j<N;++j)state[j]=initial[offset+i*N+j];
    __shared__ float r[CHUNK][N],w[CHUNK][N],k[CHUNK][N],v[CHUNK][N],a[CHUNK][N],b[CHUNK][N];
    for(int t0=0;t0<T;t0+=CHUNK){
        __syncthreads();
        #pragma unroll
        for(int tt=0;tt<CHUNK;++tt){
            if(t0+tt<T){
                long long idx=(((long long)batch*T+t0+tt)*H+head)*N+i;
                r[tt][i]=__bfloat162float(r_[idx]);
                w[tt][i]=__expf(-0.6065306597f/(1.0f+__expf(-__bfloat162float(w_[idx]))));
                k[tt][i]=__bfloat162float(k_[idx]);v[tt][i]=__bfloat162float(v_[idx]);
                a[tt][i]=__bfloat162float(a_[idx]);b[tt][i]=__bfloat162float(b_[idx]);
            }
        }
        __syncthreads();
        for(int tt=0;tt<CHUNK && t0+tt<T;++tt){
            float sa=0.f;
            #pragma unroll
            for(int j=0;j<N;++j)sa+=state[j]*a[tt][j];
            float vi=v[tt][i],y=0.f;
            #pragma unroll
            for(int j=0;j<N;++j){
                float s=state[j];
                s=s*w[tt][j]+(sa*b[tt][j]+k[tt][j]*vi);
                y+=s*r[tt][j];state[j]=s;
            }
            long long idx=(((long long)batch*T+t0+tt)*H+head)*N+i;
            out[idx]=__float2bfloat16_rn(y);
            if(t0+tt+1==state_tokens){
                #pragma unroll
                for(int j=0;j<N;++j)final[offset+i*N+j]=state[j];
            }
        }
    }
}
std::tuple<torch::Tensor,torch::Tensor> forward_state(torch::Tensor r,torch::Tensor w,
    torch::Tensor k,torch::Tensor v,torch::Tensor a,torch::Tensor b,torch::Tensor initial,int64_t state_tokens){
    TORCH_CHECK(r.is_cuda() && r.dim()==4 && r.size(3)==N && r.size(1)>0,"Expected CUDA [B,T,H,64]");
    for(auto x:{r,w,k,v,a,b}){
        TORCH_CHECK(x.device()==r.device() && x.scalar_type()==torch::kBFloat16 &&
            x.sizes()==r.sizes() && x.is_contiguous(),"Expected matching contiguous CUDA BF16 inputs");
    }
    int B=r.size(0),T=r.size(1),H=r.size(2);
    TORCH_CHECK(state_tokens>0 && state_tokens<=T,"Invalid retained-state position");
    TORCH_CHECK(initial.device()==r.device() && initial.scalar_type()==torch::kFloat32 &&
        initial.is_contiguous() && initial.sizes()==torch::IntArrayRef({B,H,N,N}),"Invalid FP32 initial state");
    const c10::cuda::CUDAGuard guard(r.device());
    auto out=torch::empty_like(r),final=torch::empty_like(initial);
    stateful_forward<<<dim3(H,B),dim3(N),0,at::cuda::getCurrentCUDAStream()>>>(T,H,
        (bf*)r.data_ptr(),(bf*)w.data_ptr(),(bf*)k.data_ptr(),(bf*)v.data_ptr(),
        (bf*)a.data_ptr(),(bf*)b.data_ptr(),initial.data_ptr<float>(),(bf*)out.data_ptr(),final.data_ptr<float>(),state_tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out,final};
}
TORCH_LIBRARY(rosa_rwkv7_stateful,m){m.def("forward",forward_state);}
