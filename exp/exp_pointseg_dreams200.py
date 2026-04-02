from exp.exp_pointseg import Exp_PointSeg
from data_provider.data_factory_dreams200 import data_provider_dreams200


class Exp_PointSeg_Dreams200(Exp_PointSeg):
    """
    复用 Exp_PointSeg 的训练/验证/测试逻辑，
    仅替换数据读取入口，以适配 preprocess_dreams.py 生成的新数据格式。
    """

    def _get_data(self, flag):
        return data_provider_dreams200(self.args, flag)
