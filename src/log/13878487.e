2021-10-15 11:43:49.540006: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.summary API due to missing TensorBoard installation.
2021-10-15 11:47:53.050029: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
2021-10-15 11:47:53.108569: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:86:00.0 name: A100-PCIE-40GB computeCapability: 8.0
coreClock: 1.41GHz coreCount: 108 deviceMemorySize: 39.59GiB deviceMemoryBandwidth: 1.41TiB/s
2021-10-15 11:47:53.108618: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-10-15 11:47:53.214264: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-10-15 11:47:53.214350: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2021-10-15 11:47:53.279920: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2021-10-15 11:47:53.307830: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2021-10-15 11:47:53.457753: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2021-10-15 11:47:53.479890: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
2021-10-15 11:47:53.481815: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2021-10-15 11:47:53.484118: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2021-10-15 11:47:53.486075: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:86:00.0 name: A100-PCIE-40GB computeCapability: 8.0
coreClock: 1.41GHz coreCount: 108 deviceMemorySize: 39.59GiB deviceMemoryBandwidth: 1.41TiB/s
2021-10-15 11:47:53.486105: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-10-15 11:47:53.486119: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-10-15 11:47:53.486129: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2021-10-15 11:47:53.486139: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2021-10-15 11:47:53.486148: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2021-10-15 11:47:53.486157: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2021-10-15 11:47:53.486166: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
2021-10-15 11:47:53.486174: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2021-10-15 11:47:53.488301: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2021-10-15 11:47:53.488331: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-10-15 11:47:54.749648: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-10-15 11:47:54.749695: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0 
2021-10-15 11:47:54.749709: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N 
2021-10-15 11:47:54.753104: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 37598 MB memory) -> physical GPU (device: 0, name: A100-PCIE-40GB, pci bus id: 0000:86:00.0, compute capability: 8.0)
2021-10-15 11:47:55.943382: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-10-15 11:47:55.943905: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2700000000 Hz
2021-10-15 11:47:57.308668: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-10-15 11:47:58.149055: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2021-10-15 11:47:58.164327: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2021-10-15 11:48:02.364144: I tensorflow/stream_executor/cuda/cuda_blas.cc:1838] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2021-10-15 11:48:07.830106: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
2021-10-15 11:49:15.029611: W tensorflow/core/util/tensor_slice_reader.cc:95] Could not open ./tmp/checkpoint: Failed precondition: tmp/checkpoint; Is a directory: perhaps your file is in a different file format and you need to use a different restore operator?
Traceback (most recent call last):
  File "main.py", line 58, in <module>
    eval(cmd())
  File "<string>", line 1, in <module>
  File "main.py", line 21, in nofair
    result = exp(data_path, fair=False)
  File "main.py", line 14, in exp
    result = experiment.exp(fairbalance = fair)
  File "/home/zxyvse/image_fairness/src/experiment.py", line 158, in exp
    self.model.fit(self.X[train], y, sample_weight=sample_weight)
  File "/home/zxyvse/image_fairness/src/cnn.py", line 109, in fit
    self.model.load_weights(checkpoint_filepath)
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-9.3.0/py-tensorflow-2.4.1-lklqe3uouvytyczpeid7tjupv3tvf3aq/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py", line 2227, in load_weights
    with h5py.File(filepath, 'r') as f:
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-9.3.0/py-h5py-2.10.0-p7fds677uu2eimxvzccxjlljqr5avh23/lib/python3.8/site-packages/h5py/_hl/files.py", line 406, in __init__
    fid = make_fid(name, mode, userblock_size,
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-9.3.0/py-h5py-2.10.0-p7fds677uu2eimxvzccxjlljqr5avh23/lib/python3.8/site-packages/h5py/_hl/files.py", line 173, in make_fid
    fid = h5f.open(name, flags, fapl=fapl)
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "h5py/h5f.pyx", line 88, in h5py.h5f.open
OSError: Unable to open file (file read failed: time = Fri Oct 15 11:49:15 2021
, filename = './tmp/checkpoint', file descriptor = 23, errno = 21, error message = 'Is a directory', buf = 0x7fffbbfb0078, total read size = 8, bytes this sub-read = 8, bytes actually read = 18446744073709551615, offset = 0)
