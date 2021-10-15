2021-10-15 09:04:04.596702: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.summary API due to missing TensorBoard installation.
Traceback (most recent call last):
  File "main.py", line 58, in <module>
    eval(cmd())
  File "<string>", line 1, in <module>
  File "main.py", line 21, in nofair
    result = exp(data_path, fair=False)
  File "main.py", line 11, in exp
    experiment = Experiment(data_path)
  File "/home/zxyvse/image_fairness/src/experiment.py", line 15, in __init__
    self.model = VGG16()
  File "/home/zxyvse/image_fairness/src/cnn.py", line 112, in __init__
    self.model = tf.keras.applications.VGG16(
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-9.3.0/py-tensorflow-2.4.1-lklqe3uouvytyczpeid7tjupv3tvf3aq/lib/python3.8/site-packages/tensorflow/python/keras/applications/vgg16.py", line 124, in VGG16
    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
ValueError: If using `weights` as `"imagenet"` with `include_top` as true, `classes` should be 1000
