from torch.utils.data import DataLoader

from data_provider.data_loader_dreams200 import DREAMSScoring1200HzSegLoader


def data_provider_dreams200(args, flag):
    """
    专用于 preprocess_dreams.py 生成数据的 data_provider。
    """
    flag_lower = flag.lower() if isinstance(flag, str) else flag
    if args.task_name != "point_segmentation":
        raise ValueError("data_provider_dreams200 currently only supports task_name='point_segmentation'")

    data_set = DREAMSScoring1200HzSegLoader(
        root_path=args.root_path,
        flag=flag_lower,
    )
    shuffle_flag = (flag_lower == "train")
    drop_last = False
    batch_size = args.batch_size

    print(flag_lower, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
