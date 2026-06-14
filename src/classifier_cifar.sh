#把/home/xxx/aet-cnn-data替换为文件aet-cnn-data.tar.xz解压后所在的目录
#-usemtcs表示用gpu训练，否则用cpu训练。
#引入libaet.so libaet_cuda.so
#export LD_LIBRARY_PATH=aet编译器安装路径/lib64:$LD_LIBRARY_PATH
./aitest classifier train -data /home/xxx/aet-cnn-data/data/classifier/cifar/cifar.data \
-cfg /home/xxx/aet-cnn-data/cfg/classifier/cifar/cifar_small.cfg   \
-usemtcs

#下面训练后的验证
#./aitest classifier valid -data /home/xxx/aet-cnn-data/data/classifier/cifar/cifar.data \
#-cfg /home/xxx/aet-cnn-data/cfg/classifier/cifar/cifar_small.cfg  \
#-weight /home/xxx/cifar_small.weights \
#-usemtcs
