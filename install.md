进入auto_eval 安装 pip install -e .
同理ecot和libero openvla


vllm

git clone https://github.com/kevinDuan1/vllm.git
cd vllm
git checkout v0.7

export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/8d/cf/9b775a1a1f5fe2f6c2d321396ad41b9849de2c76fa46d78e6294ea13be91/vllm-0.7.3-cp38-abi3-manylinux1_x86_64.whl
pip install --editable ../vllm

注意要下载已经编译好的轮子


注意flashattn 这个要编译
MAX_JOBS=4 pip install flash-attn --no-build-isolation




<!-- 

if ($?PYTHONPATH) then
    setenv PYTHONPATH /cs/student/projects1/rai/2024/jiaqiyao/pkg/lib/python3.9/site-packages:$PYTHONPATH
else
    setenv PYTHONPATH /cs/student/projects1/rai/2024/jiaqiyao/pkg/lib/python3.9/site-packages
endif -->