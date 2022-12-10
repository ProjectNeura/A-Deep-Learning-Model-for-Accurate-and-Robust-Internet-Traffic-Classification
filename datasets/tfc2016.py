import binascii
import numpy as np
from PIL import Image
from papercandy import *
from torch import Tensor
from random import shuffle
from os.path import exists
from os import listdir, mkdir
from scapy.plist import PacketList
from scapy.all import rdpcap as rdpcap
from torchvision.transforms import Compose, Resize, ToTensor


class Pcap(object):
    def __init__(self, packets: PacketList, name: Union[str, None] = None):
        self.packets: PacketList = packets
        self.name: Union[str, None] = name

    def __len__(self) -> int:
        return len(self.packets)

    def raw_list(self) -> list[bytes]:
        r = []
        for packet in self.packets:
            r.append(packet.original)
        return r


def read_pcap(filename: Union[str, PathLike], name: Union[str, None] = None):
    return Pcap(rdpcap(filename), name)


def fill_padding(array: list, width: int) -> np.ndarray:
    while len(array) < width:
        array += array
    res = []
    lines = array.copy()
    for _ in range(width):
        ll = len(lines)
        if ll <= width:
            res.append(np.asarray(lines + array[:width - ll]))
            lines = array[width - ll:]
            continue
        res.append(np.asarray(lines[:width]))
        lines = lines[width:]
    return np.asarray(res)


def bytes2array(content: bytes, width: int) -> np.ndarray:
    hex_content = binascii.hexlify(content)
    fh = [int(hex_content[i: i + 2], 16) for i in range(0, len(hex_content), 2)]
    fh = fill_padding(fh, width)
    fh = np.reshape(fh, (width, width))
    fh = np.uint8(fh)
    return fh


def array2img(array: np.ndarray):
    return Image.fromarray(array)


CLASSES = {
    "bittorrent": 0,
    "cridex": 1,
    "facetime": 2,
    "ftp": 3,
    "geodo": 4,
    "htbot": 5,
    "miuref": 6,
    "mysql": 7,
    "neris": 8,
    "outlook": 9,
    "skype": 10,
    "tinba": 11,
    "zeus": 12,
}


def name2id(name: str) -> int:
    return CLASSES[name]


def to_img(src: str, dst: str, width: int = 28):
    for filename in listdir(src):
        group = read_pcap(f"{src}/{filename}", filename[:-5].lower())
        group_dst = f"{dst}/{group.name}"
        if exists(group_dst):
            continue
        mkdir(group_dst)
        counter = 0
        for raw in group.raw_list():
            array2img(bytes2array(raw, width)).save(f"{group_dst}/{counter}.png")
            counter += 1


class TFC2016(Dataset):
    """
    What is happening here is a conversion between bytes and image.
    """

    def __init__(self, src: Union[str, PathLike], width: int = 28):
        self.src: str = src
        self.group_list: list[Pcap] = []
        self.dc_list: list[DataCompound] = []

        for filename in listdir(self.src):
            self.group_list.append(read_pcap(f"{self.src}/{filename}", filename[:-5].lower()))
        for group in self.group_list:
            transform = Compose([Resize((width, width)), ToTensor()])
            self.dc_list += [DataCompound(transform(array2img(bytes2array(raw, width))).unsqueeze(0), Tensor([name2id(group.name)])) for raw in
                             group.raw_list()]

    def __len__(self) -> int:
        return len(self.dc_list)

    def cut(self, i: slice) -> Self:
        raise RuntimeError("Doesn't support cutting.")

    def get(self, i: int) -> DataCompound:
        return self.dc_list[i]


class TFC2016Image(Dataset):
    class Image(object):
        def __init__(self, filename: str, group_id: int):
            self.filename: str = filename
            self.group_id: int = group_id

    def __init__(self, src: Union[str, PathLike], width: int = 224):
        self.src: str = src
        self.image_list: list[TFC2016Image.Image] = []
        self.width: int = width
        self.load()

    def __len__(self) -> int:
        return len(self.image_list)

    def load(self):
        for group in listdir(self.src):
            gfn = f"{self.src}/{group}"
            for image in listdir(gfn):
                self.image_list.append(TFC2016Image.Image(f"{gfn}/{image}", name2id(group)))

    def shuffle(self) -> Self:
        shuffle(self.image_list)
        return self

    def cut(self, i: slice) -> Self:
        self.image_list = self.image_list[i]
        return self

    def get(self, i: int) -> DataCompound:
        image_info = self.image_list[i]
        img = Image.open(image_info.filename)
        transform = Compose([Resize((self.width, self.width)), ToTensor()])
        img = transform(img)
        return DataCompound(img, Tensor([image_info.group_id]))


class ClassificationDataloader(Dataloader):
    def combine_batch(self, data_batch: list[DataCompound]) -> DataCompound:
        d = super(ClassificationDataloader, self).combine_batch(data_batch)
        d.target = d.target.reshape(d.target.shape[0]).long()
        return d
