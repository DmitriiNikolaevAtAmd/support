from nemo.collections import llm
import nemo_run as run

def run_pretrain():
    # Use Mistral 7B model
    recipe = llm.mistral_7b.pretrain_recipe(
        name="mistral_7b_pretrain_fp8",
        dir="/checkpoints",
        num_nodes=1,
        num_gpus_per_node=8,
    )
    
    recipe.data.micro_batch_size = 1
    recipe.data.global_batch_size = 8
    recipe.trainer.max_steps = 10
    
    recipe.model.config.fp8 = "hybrid"  
    recipe.model.config.fp8_param = True
    
    # Use 4 for tensor parallel (32 heads / 4 = 8 heads per GPU)
    # Use 2 for pipeline parallel to utilize all 8 GPUs (4 * 2 = 8)
    recipe.trainer.strategy.tensor_model_parallel_size = 4
    recipe.trainer.strategy.pipeline_model_parallel_size = 2
    
    recipe.trainer.enable_checkpointing = False
    recipe.resume = None
    
    run.run(recipe, direct=True)

if __name__ == "__main__":
    run_pretrain()

