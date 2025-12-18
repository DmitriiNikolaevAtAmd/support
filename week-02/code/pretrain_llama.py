from nemo.collections import llm
import nemo_run as run

def run_pretrain():
    recipe = llm.llama31_8b.pretrain_recipe(
        name="llama3_1_8b_pretrain_fp8",
        dir="/checkpoints",
        num_nodes=1,
        num_gpus_per_node=8,
    )
    
    # MINIMAL batch sizes
    recipe.data.micro_batch_size = 1
    recipe.data.global_batch_size = 8
    
    # Very short training
    recipe.trainer.max_steps = 10  # Reduced from 50
    
    # FP8 to save memory
    recipe.model.config.fp8 = "hybrid"  
    recipe.model.config.fp8_param = True
    
    # Tensor parallelism to split model across GPUs
    recipe.trainer.strategy.tensor_model_parallel_size = 4  # Increased from 2
    recipe.trainer.strategy.pipeline_model_parallel_size = 1
    
    # NO activation checkpointing for now (simplify)
    # Just use defaults
    
    # DISABLE CHECKPOINTING
    recipe.trainer.enable_checkpointing = False
    
    # Disable resume
    recipe.resume = None
    
    run.run(recipe, direct=True)

if __name__ == "__main__":
    run_pretrain()

