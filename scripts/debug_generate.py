import torch
import gc
import os
import sys
import pdb

# 设置路径
sys.path = [p for p in sys.path if 'triper-project' not in p]
sys.path.append('/home/wly/szl_all_code/triper-project')

# 清理缓存
torch.cuda.empty_cache()
gc.collect()

def debug_triper_generate():
    """调试Triper生成过程"""
    
    print("🔧 开始调试Triper生成过程...")
    
    # 加载模型
    from triper.model import from_pretrained_components
    from triper.data import TriperDataset, TriperDataCollator

    audio_config = {
        'mm_audio_encoder': 'whisper_vq',
        'audio_hidden_size': 1280,
        'audio_model_path': '/sda1/glm-4-voice-tokenizer',
        'audio_projector_type': 'mlp2x_gelu',
        'audio_projector_hidden_dim': 2048,
        'dropout': 0.1
    }

    print("📦 加载模型组件...")
    tokenizer, triper_model, image_processor, context_len, audio_encoder = from_pretrained_components(
        llava_model_path="/sda1/llava-v1.5-13b",
        audio_encoder_path="/sda1/glm-4-voice-tokenizer",
        audio_projector_path=None,
        audio_config=audio_config,
        freeze_llava=True,
        device_map="cuda:3"
    )

    print("📊 加载数据...")
    dataset = TriperDataset(
        json_path='/home/wly/szl_all_code/triper-project/data/simple_data_20_samples.json',
        media_root_path='/home/wly/szl_all_code/triper-project/data',
    )

    collator = TriperDataCollator(
        tokenizer=tokenizer,
        image_processor=image_processor,
        audio_processor=audio_encoder,
        model_cfg=triper_model.llava_model.config
    )

    # 准备单个样本进行调试
    print("🧪 准备单个样本...")
    single_sample = [dataset[0]]
    single_batch = collator(single_sample)

    # 移动到设备
    device_batch = {}
    for k, v in single_batch.items():
        if hasattr(v, 'to'):
            device_batch[k] = v.to(triper_model.device)
        else:
            device_batch[k] = v

    print(f"✅ 数据准备完成:")
    print(f"  input_ids: {device_batch['input_ids'].shape}")
    print(f"  attention_mask: {device_batch['attention_mask'].shape}")
    print(f"  images: {device_batch['images'].shape}")
    print(f"  audio_features: {device_batch['audio_features'].shape}")


    # 🔧 设置断点 - 在调用generate之前
    print("🚨 设置断点1: 调用triper_model.generate之前")
    pdb.set_trace()
    
    # 在pdb中你可以检查:
    # - device_batch 的内容
    # - triper_model 的状态
    # - 各种参数的值
    
    try:
        print("🎯 开始调用triper_model.generate...")
        
        # 🔧 在generate方法内部也设置断点
        # 我们需要修改generate方法来添加更多断点
        original_generate = triper_model.generate
        
        def debug_generate(*args, **kwargs):
            print("🚨 设置断点2: 进入generate方法内部")
            pdb.set_trace()
            # 在这里你可以检查传入的参数
            
            return original_generate(*args, **kwargs)
        
        # 临时替换generate方法
        triper_model.generate = debug_generate
        
        response = triper_model.generate(
            input_ids=device_batch['input_ids'],
            attention_mask=device_batch['attention_mask'],
            images=device_batch['images'],
            audio_features=device_batch['audio_features'],
            max_new_tokens=5,  # 用很少的token便于调试
            temperature=0.1,
            do_sample=False,  # 贪心搜索，确定性结果
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        print(f"✅ 生成成功: {response.shape}")
        
        # 解码结果
        original_len = device_batch['input_ids'].shape[1]
        if response.shape[1] > original_len:
            generated_part = response[0, original_len:]
            generated_text = tokenizer.decode(generated_part, skip_special_tokens=True)
            print(f"新生成的文本: '{generated_text}'")
        else:
            generated_text = tokenizer.decode(response[0], skip_special_tokens=True)
            print(f"生成的文本: '{generated_text}'")

    except Exception as e:
        print(f"❌ 错误发生: {e}")
        print(f"错误类型: {type(e).__name__}")
        
        # 🔧 设置断点3: 错误发生时
        print("🚨 设置断点3: 错误发生时")
        pdb.set_trace()
        
        # 在这里你可以检查:
        # - 错误的详细信息
        # - 当前的变量状态
        # - 调用栈
        
        import traceback
        traceback.print_exc()
        raise

def debug_llava_methods():
    """专门调试LLaVA的相关方法"""
    
    print("🔍 调试LLaVA方法调用...")
    
    # 首先加载必要组件（简化版）
    from triper.model import from_pretrained_components
    
    print("📦 快速加载模型...")
    tokenizer, triper_model, image_processor, context_len, audio_encoder = from_pretrained_components(
        llava_model_path="/sda1/llava-v1.5-13b",
        audio_encoder_path="/sda1/glm-4-voice-tokenizer",
        audio_projector_path=None,
        audio_config={'mm_audio_encoder': 'whisper_vq', 'audio_hidden_size': 1280},
        freeze_llava=True,
        device_map="cuda:3"
    )
    
    # 检查LLaVA模型的方法
    print("🔍 检查LLaVA模型方法:")
    llava_model = triper_model.llava_model
    print(f"  类型: {type(llava_model).__name__}")
    print(f"  有generate方法: {hasattr(llava_model, 'generate')}")
    print(f"  有prepare_inputs_labels_for_multimodal: {hasattr(llava_model, 'prepare_inputs_labels_for_multimodal')}")
    print(f"  有prepare_inputs_for_generation: {hasattr(llava_model, 'prepare_inputs_for_generation')}")
    
    # 🔧 设置断点4: 检查LLaVA模型状态
    print("🚨 设置断点4: 检查LLaVA模型状态")
    pdb.set_trace()
    
    # 在这里你可以:
    # - 查看llava_model的所有属性和方法
    # - 检查llava_model.config
    # - 查看模型的内部结构

if __name__ == "__main__":
    print("🎯 Triper生成调试脚本")
    print("=" * 50)
    
    try:
        # 选择调试模式
        print("选择调试模式:")
        print("1. 调试完整生成过程 (debug_triper_generate)")
        print("2. 调试LLaVA方法 (debug_llava_methods)")
        
        choice = input("请输入选择 (1/2): ").strip()
        
        if choice == "1":
            debug_triper_generate()
        elif choice == "2":
            debug_llava_methods()
        else:
            print("默认运行完整调试...")
            debug_triper_generate()
            
    except KeyboardInterrupt:
        print("\n🛑 调试被用户中断")
    except Exception as e:
        print(f"\n❌ 调试过程出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 调试完成")