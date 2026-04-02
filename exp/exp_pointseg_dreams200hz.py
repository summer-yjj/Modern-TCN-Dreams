from exp.exp_pointseg import Exp_PointSeg
from data_provider.data_factory_dreams200 import data_provider_dreams200


class Exp_PointSeg_Dreams200Hz(Exp_PointSeg):
    """
    200Hz DREAMS 适配实验类：
    复用 Exp_PointSeg 训练/验证/测试逻辑，只替换数据读取入口。
    """

    def _get_data(self, flag):
        return data_provider_dreams200(self.args, flag)
