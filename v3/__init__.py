try:
	# Import submodules as modules to keep live references (avoid `from ... import *`)
	from . import helper
	from .predict import *
	from .video import *
	from .train import train
except Exception:
	# Fallback for environments where package context still isn't available
	import helper
	from predict import *
	from video import *
	from train import train

# Convenience re-exports for commonly used helpers
set_model = helper.set_model
best_device = helper.best_device

def __getattr__(name: str): return getattr(helper, name)


