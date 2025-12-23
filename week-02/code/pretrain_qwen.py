from nemo.collections import llm
import nemo_run as run

def run_pretrain():
    # Use Qwen 2.5 7B model
    recipe = llm.qwen25_7b.pretrain_recipe(
        name="qwen25_7b_test_fp8",
        dir="/checkpoints",
        num_nodes=1,
        num_gpus_per_node=8,
    )
    
    recipe.data.micro_batch_size = 1
    recipe.data.global_batch_size = 8
    recipe.trainer.max_steps = 10
    
    recipe.model.config.fp8 = "hybrid"  
    recipe.model.config.fp8_param = True
    
    # Use 4 for tensor parallel (28 heads / 4 = 7 heads per GPU)
    # Use 2 for pipeline parallel to utilize all 8 GPUs (4 * 2 = 8)
    recipe.trainer.strategy.tensor_model_parallel_size = 4
    recipe.trainer.strategy.pipeline_model_parallel_size = 2
    
    recipe.trainer.enable_checkpointing = False
    recipe.resume = None
    
    run.run(recipe, direct=True)

if __name__ == "__main__":
    run_pretrain()
