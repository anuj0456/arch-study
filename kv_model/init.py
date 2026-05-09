from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from kv_model.config import KiteFishConfig
from kv_model.model import KiteFishModel, KiteFishForCausalLM



AutoConfig.register("kitefish_v1", KiteFishConfig)
AutoModel.register(KiteFishConfig, KiteFishModel)
AutoModelForCausalLM.register(KiteFishConfig, KiteFishForCausalLM)