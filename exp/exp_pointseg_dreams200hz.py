from exp.exp_pointseg_dreams200 import Exp_PointSeg_Dreams200


class Exp_PointSeg_Dreams200Hz(Exp_PointSeg_Dreams200):
    """
    200Hz DREAMS 适配实验类（别名类）：
    继承 Exp_PointSeg_Dreams200 的数据读取与训练流程，
    并共享 Exp_PointSeg 中新增的 CSV 指标导出与运行日志导出。
    """
    pass
