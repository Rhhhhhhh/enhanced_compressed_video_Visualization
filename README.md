# enhanced_compressed_video_Visualization

To visualizing enhanced compressed video (mfqev2 dataset)

根据[stdf](https://github.com/RyanXingQL/STDF-PyTorch)代码中的test_one_video.py修改而来。

其实是我自己不会可视化单帧，我看大部分人包括我师兄都是先转rgb再转图片，我又懒得看那些代码，只好自己用winhex看了一下yuv文件的存储格式。

由于mfqev2数据集都是8bit的420视频，因此保存时是以每个像素都有一个y值，每4个像素一个u和一个v（是的，420是4y1u1v而不是2u0v），写入的时候是顺序写入：例如一个720\*480 的视频，那么就是有720\*480个y通道的像素，720\*480/4个u通道和v通道像素，第一帧就是720\*480+720\*480/4\*2个0-255的值，顺序写入即可。

10bit和非420的视频格式我没了解过，代码不出意外是不可以用的。
