import comfy.model_management as model_management
import gc
import torch
import weakref


# Note: This doesn't work with reroute for some reason?
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

class ForceUnloadModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "passthrough0": (any, ),
                "passthrough1": (any, ),
                "passthrough2": (any, ),
                "passthrough3": (any, ),
                "passthrough4": (any, ),
                "passthrough5": (any, ),
                "passthrough6": (any, ),
                "passthrough7": (any, ),
                "passthrough8": (any, ),
                "passthrough9": (any, ),
            },
        }
    
    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True
    
    RETURN_TYPES = (any, any, any, any, any, any, any, any, any, any)
    FUNCTION = "route"
    CATEGORY = "Unload Model"
    DESCRIPTION = "Unload all models except those that are passing through this node.\nCombine with \"--cache-none\" argument to achive best results.\nUse \"passthrough\" inputs/outputs to force correct node execution order. Value passed over this node is unchanged."

    @classmethod
    def _any_model(s, container: set, model):
        if model is not None:
            container.add(id(model))
        if isinstance(model, weakref.ref):
            ForceUnloadModels._any_model(container, model())
        if hasattr(model, "model"):
            ForceUnloadModels._any_model(container, model.model)
        if hasattr(model, "real_model"):
            ForceUnloadModels._any_model(container, model.real_model)
        return container
    
    @classmethod
    def _get_keep_loaded(s, models: tuple):
        # Create keep_loaded list that contains models that should not be unloaded (they are passing through this node)
        keep_models_full_set = set()
        for m in models:
            ForceUnloadModels._any_model(keep_models_full_set, m)
        keep_loaded = []
        for m in model_management.current_loaded_models:
            inner_models = ForceUnloadModels._any_model(set(), m)
            if len(inner_models.intersection(keep_models_full_set)) != 0:
                keep_loaded.append(m)
        return keep_loaded

    def route(self, **kwargs):
        result = (kwargs["passthrough0"] if "passthrough0" in kwargs else None,
                kwargs["passthrough1"] if "passthrough1" in kwargs else None,
                kwargs["passthrough2"] if "passthrough2" in kwargs else None,
                kwargs["passthrough3"] if "passthrough3" in kwargs else None,
                kwargs["passthrough4"] if "passthrough4" in kwargs else None,
                kwargs["passthrough5"] if "passthrough5" in kwargs else None,
                kwargs["passthrough6"] if "passthrough6" in kwargs else None,
                kwargs["passthrough7"] if "passthrough7" in kwargs else None,
                kwargs["passthrough8"] if "passthrough8" in kwargs else None,
                kwargs["passthrough9"] if "passthrough9" in kwargs else None)

        keep_loaded = ForceUnloadModels._get_keep_loaded(result)
        initial_loaded_models = len(model_management.current_loaded_models)
        initial_keep_loaded_count = len(keep_loaded)

        # First, try to unload using standard model_management API
        unloaded_list = model_management.free_memory(1e30, model_management.get_torch_device(), keep_loaded)
        unloaded_count = len(unloaded_list)
        print(f"Unloaded {unloaded_count} models using model_management.free_memory")

        # If there is still models loaded, we will try to unload them manually
        keep_loaded = ForceUnloadModels._get_keep_loaded(result)
        loadedmodels = model_management.current_loaded_models
        if len(loadedmodels) > 0 and len(loadedmodels) > len(keep_loaded):
            index = 0
            while index < len(loadedmodels):
                m = loadedmodels[index]
                if m in keep_loaded:
                    index += 1
                else:
                    m.model_unload()
                    del m
                    del loadedmodels[index]
                    unloaded_count += 1
        print(f"Total unloaded models {unloaded_count} of {initial_loaded_models} (keeping {initial_keep_loaded_count} model required)")

        # Clear caches and GC, if anything was unloaded
        if unloaded_count > 0:
            model_management.soft_empty_cache()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()

        # Return the original inputs, so the node can be used in a reroute
        return result

NODE_CLASS_MAPPINGS = {
    "ForceUnloadModels": ForceUnloadModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ForceUnloadModels": "Force Unload All Models",
}
