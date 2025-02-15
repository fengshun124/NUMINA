import torch
import torch.distributed as dist
from utils.distributed import get_rank, is_dist_avail_and_initialized, is_main_process
import random
import logging

logger = logging.getLogger(__name__)

# MetaLoader 类是一个封装多个数据加载器（dataloader）的工具。它的设计目的是在分布式或多进程训练环境中，从多个数据加载器中按一定顺序迭代批次（batch）。
class MetaLoader(object):
    """ wraps multiple data loader """
    def __init__(self, name2loader):
        """Iterates over multiple dataloaders, it ensures all processes
        work on data from the same dataloader. This loader will end when
        the shorter dataloader raises StopIteration exception.

        loaders: Dict, {name: dataloader}
        """
        self.name2loader = name2loader
        
        # 将每个加载器转换为对应的迭代器
        self.name2iter = {name: iter(l) for name, l in name2loader.items()}  
        
        # 为每个加载器分配一个唯一的整数索引，方便后续操作
        name2index = {name: idx for idx, (name, l) in enumerate(name2loader.items())}
        index2name = {v: k for k, v in name2index.items()}

        # 按加载器的长度（len(l)）为每个加载器生成索引，并组合成一个总的迭代顺序列表
        # 例如，如果加载器 A 长度为 3，加载器 B 长度为 2，那么初始顺序为 [0, 0, 0, 1, 1]
        iter_order = []
        for n, l in name2loader.items():
            iter_order.extend([name2index[n]]*len(l))

        # 随机打乱迭代顺序
        random.shuffle(iter_order)
        iter_order = torch.Tensor(iter_order).to(torch.device("cuda")).to(torch.uint8)

        # sync
        # 如果启用了分布式训练，则通过 dist.broadcast 将第一个进程的 iter_order 广播给其他进程，确保所有进程的迭代顺序一致
        if is_dist_avail_and_initialized():
            # make sure all processes have the same order so that
            # each step they will have data from the same loader
            dist.broadcast(iter_order, src=0)
            
        # 将 iter_order 中的索引映射回加载器名称，得到最终的迭代顺序
        self.iter_order = [index2name[int(e.item())] for e in iter_order.cpu()]

        logger.info(str(self))

    def __str__(self):
        output = [f"MetaLoader has {len(self.name2loader)} dataloaders, {len(self)} batches in total"]
        for idx, (name, loader) in enumerate(self.name2loader.items()):
            output.append(
                f"dataloader index={idx} name={name}, batch-size={loader.batch_size} length(#batches)={len(loader)} "
            )
        return "\n".join(output)

    def __len__(self):
        return len(self.iter_order)

    def __iter__(self):
        """ this iterator will run indefinitely """
        for name in self.iter_order:
            _iter = self.name2iter[name]
            batch = next(_iter)
            yield name, batch
