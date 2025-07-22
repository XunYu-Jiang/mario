import os
import torch
import logging, log_setting

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(log_setting.MyFormatter())
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


logger.debug(os.environ['CONDA_DEFAULT_ENV'])
logger.info(torch.__version__)
logger.warning(torch.cuda.is_available())
logger.error(torch.version.cuda)
logger.critical(torch.backends.cudnn.version())
print("test")