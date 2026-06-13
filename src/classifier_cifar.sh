#把/home/xxx/aet-cnn-data替换为文件aet-cnn-data.tar.xz解压后所在的目录
#-usemtcs表示用gpu训练，否则用cpu训练。
./aitest classifier train -data /home/xxx/aet-cnn-data/cfg/classifier/cifar/cifar.data \
-cfg /home/xxx/aet-cnn-data/cfg/classifier/cifar/cifar_small.cfg   \
-usemtcs

#./aitest classifier valid -data /home/xxx/aet-cnn-data/cfg/classifier/cifar/cifar.data \
#-cfg /home/xxx/aet-cnn-data/cfg/classifier/cifar/cifar_small.cfg  -weight /home/sns/darknet-resource/backup/classfier/cifar/aet.weights \
#-usemtcs
