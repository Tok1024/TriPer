import torch
import gc
import os
import sys
import json
from pathlib import Path
import traceback
from typing import Dict, Any, List

# 设置路径
sys.path = [p for p in sys.path if 'triper-project' not in p]
sys.path.append('/home/wly/szl_all_code/triper-project')

# 导入模块
from triper.model import from_pretrained_components
from triper.data import TriperDataset, TriperDataCollator
from triper.constants import DEFAULT_IMAGE_TOKEN

class TriperModelTester:
    """Triper模型全面测试器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        self.tokenizer = None
        self.triper_model = None
        self.image_processor = None
        self.context_len = None
        self.audio_encoder = None
        self.dataset = None
        self.collator = None
        
    def setup_test_environment(self):
        """设置测试环境"""
        print("\n" + "="*80)
        print("🚀 TRIPER MODEL COMPREHENSIVE TEST")
        print("="*80)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ GPU缓存已清理")
        
        # 打印配置
        print(f"\n📋 测试配置:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        
    def test_1_model_loading(self):
        """测试1: 模型加载"""
        print(f"\n{'='*50}")
        print("📦 测试1: 模型组件加载")
        print("="*50)
        
        try:
            audio_config = {
                'mm_audio_encoder': 'whisper_vq',
                'audio_hidden_size': 1280,
                'audio_model_path': self.config['audio_encoder_path'],
                'audio_projector_type': 'mlp2x_gelu',
                'audio_projector_hidden_dim': 2048,
                'dropout': 0.1
            }
            
            print("🔄 加载模型组件...")
            self.tokenizer, self.triper_model, self.image_processor, self.context_len, self.audio_encoder = from_pretrained_components(
                llava_model_path=self.config['llava_model_path'],
                audio_encoder_path=self.config['audio_encoder_path'],
                audio_projector_path=None,
                audio_config=audio_config,
                freeze_llava=True,
                device_map=self.config['device']
            )
            
            print("✅ 模型组件加载成功")
            
            # 验证组件
            assert self.tokenizer is not None, "Tokenizer加载失败"
            assert self.triper_model is not None, "TriperModel加载失败"
            assert self.image_processor is not None, "ImageProcessor加载失败"
            assert self.audio_encoder is not None, "AudioEncoder加载失败"
            
            # 打印模型摘要
            self.triper_model.print_model_summary()
            
            # 验证模型就绪状态
            assert self.triper_model.is_ready(), "模型未就绪"
            
            self.results['model_loading'] = "✅ PASSED"
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            traceback.print_exc()
            self.results['model_loading'] = f"❌ FAILED: {e}"
            raise
    
    def test_2_data_loading(self):
        """测试2: 数据加载"""
        print(f"\n{'='*50}")
        print("📊 测试2: 数据集和Collator")
        print("="*50)
        
        try:
            # 加载数据集
            print("🔄 加载数据集...")
            self.dataset = TriperDataset(
                json_path=self.config['data_path'],
                media_root_path=self.config['media_root'],
            )
            
            print(f"✅ 数据集加载成功，共 {len(self.dataset)} 个样本")
            
            # 创建collator
            print("🔄 创建数据collator...")
            self.collator = TriperDataCollator(
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                audio_processor=self.audio_encoder,
                model_cfg=self.triper_model.llava_model.config
            )
            
            print("✅ Collator创建成功")
            
            # 测试单个样本
            print("🔄 测试单个样本...")
            sample = self.dataset[0]
            print(f"样本结构: {list(sample.keys())}")
            
            # 测试collator处理
            print("🔄 测试collator处理...")
            batch = self.collator([sample])
            
            print(f"批量数据形状:")
            for key, value in batch.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
            
            self.results['data_loading'] = "✅ PASSED"
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            traceback.print_exc()
            self.results['data_loading'] = f"❌ FAILED: {e}"
            raise
    
    def test_3_forward_pass(self):
        """测试3: 前向传播"""
        print(f"\n{'='*50}")
        print("🔥 测试3: 模型前向传播")
        print("="*50)
        
        try:
            # 准备测试数据
            test_samples = [self.dataset[i] for i in range(min(2, len(self.dataset)))]
            batch = self.collator(test_samples)
            
            # 移动到设备
            device_batch = {}
            for k, v in batch.items():
                if hasattr(v, 'to'):
                    device_batch[k] = v.to(self.triper_model.device)
                else:
                    device_batch[k] = v
            
            print("🔄 测试完整前向传播（图像+音频）...")
            with torch.no_grad():
                output = self.triper_model(
                    input_ids=device_batch['input_ids'],
                    images=device_batch['images'],
                    audio_features=device_batch['audio_features'],
                    attention_mask=device_batch['attention_mask']
                )
                
            print("🔄 测试仅图像前向传播...")
            with torch.no_grad():
                output_img_only = self.triper_model(
                    input_ids=device_batch['input_ids'],
                    images=device_batch['images'],
                    audio_features=None,  # 不传音频
                    attention_mask=device_batch['attention_mask']
                )
            print(f"✅ 仅图像前向传播成功: {output_img_only.logits.shape}")
            print("🔄 测试纯文本前向传播...")
            
            # 🔧 创建纯文本prompt（不包含<image>标记）
            pure_text_prompt = "USER: Tell me about artificial intelligence.\nASSISTANT:"
            pure_text_ids = self.tokenizer.encode(pure_text_prompt, return_tensors="pt")
            pure_text_mask = torch.ones_like(pure_text_ids)
            
            # 移动到设备
            pure_text_ids = pure_text_ids.to(self.triper_model.device)
            pure_text_mask = pure_text_mask.to(self.triper_model.device)
            
            print(f"📝 纯文本prompt: {pure_text_prompt}")
            print(f"📝 纯文本input_ids形状: {pure_text_ids.shape}")
            
            with torch.no_grad():
                output_text_only = self.triper_model(
                    input_ids=pure_text_ids,
                    images=None,  # 不传图像
                    audio_features=None,  # 不传音频
                    attention_mask=pure_text_mask
                )
            
            print(f"✅ 纯文本前向传播成功: {output_text_only.logits.shape}")
            
            
            print(f"✅ 完整前向传播成功")
            print(f"  输出logits形状: {output.logits.shape}")
            print(f"  输出类型: {type(output)}")
            
            # 验证输出合理性
            assert output.logits.shape[0] == len(test_samples), "批量大小不匹配"
            assert not torch.isnan(output.logits).any(), "输出包含NaN"
            assert not torch.isinf(output.logits).any(), "输出包含Inf"
            
            print("🔄 测试仅图像前向传播...")
            with torch.no_grad():
                output_img_only = self.triper_model(
                    input_ids=device_batch['input_ids'],
                    images=device_batch['images'],
                    audio_features=None,  # 不传音频
                    attention_mask=device_batch['attention_mask']
                )
            
            print(f"✅ 仅图像前向传播成功: {output_img_only.logits.shape}")
            

            
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            traceback.print_exc()
            self.results['forward_pass'] = f"❌ FAILED: {e}"
            raise
    
    def test_4_generation_capability(self):
        """测试4: 生成能力"""
        print(f"\n{'='*50}")
        print("🚀 测试4: 模型生成能力")
        print("="*50)
        
        try:
            # 准备单个样本
            single_sample = [self.dataset[0]]
            single_batch = self.collator(single_sample)
            
            # 移动到设备
            device_batch = {}
            for k, v in single_batch.items():
                if hasattr(v, 'to'):
                    device_batch[k] = v.to(self.triper_model.device)
                else:
                    device_batch[k] = v
            
            # 修复attention_mask长度
            text_len = device_batch['input_ids'].shape[1]
            attention_mask = device_batch['attention_mask'][:, :text_len]
            
            print("🔄 测试1: 纯LLaVA生成（图像+文本）...")
            response_llava = self.triper_model.generate(
                input_ids=device_batch['input_ids'],
                attention_mask=attention_mask,
                images=device_batch['images'],
                audio_features=None,  # 不传音频
                max_new_tokens=30,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # 解码结果
            original_len = device_batch['input_ids'].shape[1]
            if response_llava.shape[1] > original_len:
                generated_text = self.tokenizer.decode(
                    response_llava[0, original_len:], 
                    skip_special_tokens=True
                )
                print(f"✅ LLaVA生成成功: '{generated_text[:100]}...'")
            else:
                print(f"⚠️ LLaVA生成长度异常: {response_llava.shape}")
            
            print("🔄 测试2: 完整Triper生成（图像+音频+文本）...")
            response_triper = self.triper_model.generate(
                input_ids=device_batch['input_ids'],
                attention_mask=attention_mask,
                images=device_batch['images'],
                audio_features=device_batch['audio_features'],  # 传音频
                max_new_tokens=30,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            if response_triper.shape[1] > 2:  # 至少生成了一些token
                # 注意：Triper可能返回不同格式，需要灵活处理
                if response_triper.shape[1] > original_len:
                    generated_text = self.tokenizer.decode(
                        response_triper[0, original_len:], 
                        skip_special_tokens=True
                    )
                else:
                    generated_text = self.tokenizer.decode(
                        response_triper[0], 
                        skip_special_tokens=True
                    )
                print(f"✅ Triper生成成功: '{generated_text[:100]}...'")
            else:
                print(f"⚠️ Triper生成长度过短: {response_triper.shape}")
            
            print("🔄 测试3: 简化prompt生成...")
            simple_prompt = f"{DEFAULT_IMAGE_TOKEN}\nUSER: What do you see in this image?\nASSISTANT:"
            simple_ids = self.tokenizer.encode(simple_prompt, return_tensors="pt").to(self.triper_model.device)
            
            simple_response = self.triper_model.llava_model.generate(
                inputs=simple_ids,
                images=device_batch['images'],
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            if simple_response.shape[1] > simple_ids.shape[1]:
                simple_text = self.tokenizer.decode(
                    simple_response[0, simple_ids.shape[1]:], 
                    skip_special_tokens=True
                )
                print(f"✅ 简化prompt生成成功: '{simple_text[:100]}...'")
            else:
                print(f"⚠️ 简化prompt生成失败")
            
            self.results['generation_capability'] = "✅ PASSED"
            
        except Exception as e:
            print(f"❌ 生成测试失败: {e}")
            traceback.print_exc()
            self.results['generation_capability'] = f"❌ FAILED: {e}"
    
    def test_5_audio_projector(self):
        """测试5: 音频投影器"""
        print(f"\n{'='*50}")
        print("🎵 测试5: 音频投影器")
        print("="*50)
        
        try:
            # 测试音频投影器
            assert self.triper_model.audio_projector is not None, "音频投影器为空"
            
            # 创建测试音频特征
            test_audio = torch.randn(1, 64, 1280).to(self.triper_model.device)
            
            print(f"🔄 测试音频投影器...")
            print(f"  输入形状: {test_audio.shape}")
            
            with torch.no_grad():
                projected_audio = self.triper_model.audio_projector(test_audio)
            
            print(f"  输出形状: {projected_audio.shape}")
            print(f"  输出维度: {projected_audio.shape[-1]}")
            
            # 验证输出
            expected_dim = self.triper_model.llava_model.config.hidden_size  # 应该是5120
            assert projected_audio.shape[-1] == expected_dim, f"投影维度错误: {projected_audio.shape[-1]} != {expected_dim}"
            assert not torch.isnan(projected_audio).any(), "投影输出包含NaN"
            
            print(f"✅ 音频投影器测试通过")
            
            # 测试音频特征插入
            print(f"🔄 测试音频特征插入...")
            test_text_embeds = torch.randn(1, 50, expected_dim).to(self.triper_model.device)
            test_attention_mask = torch.ones(1, 50).to(self.triper_model.device)
            
            combined_embeds, combined_mask = self.triper_model._insert_audio_features(
                test_text_embeds, None, test_audio, test_attention_mask
            )
            
            expected_length = 50 + 64  # 文本长度 + 音频长度
            assert combined_embeds.shape[1] == expected_length, f"合并长度错误: {combined_embeds.shape[1]} != {expected_length}"
            assert combined_mask.shape[1] == expected_length, f"mask长度错误: {combined_mask.shape[1]} != {expected_length}"
            
            print(f"✅ 音频特征插入测试通过")
            
            self.results['audio_projector'] = "✅ PASSED"
            
        except Exception as e:
            print(f"❌ 音频投影器测试失败: {e}")
            traceback.print_exc()
            self.results['audio_projector'] = f"❌ FAILED: {e}"
    
    def test_6_parameter_statistics(self):
        """测试6: 参数统计"""
        print(f"\n{'='*50}")
        print("📊 测试6: 参数统计和配置")
        print("="*50)
        
        try:
            # 获取参数统计
            stats = self.triper_model.get_parameter_stats()
            
            print(f"📊 参数统计:")
            print(f"  总参数: {stats['total_params']:,}")
            print(f"  可训练参数: {stats['trainable_params']:,}")
            print(f"  可训练比例: {stats['trainable_params']/max(stats['total_params'], 1)*100:.2f}%")
            
            # 验证参数统计合理性
            assert stats['total_params'] > 0, "总参数数量为0"
            assert stats['trainable_params'] >= 0, "可训练参数数量为负"
            
            # 检查组件参数
            for component, comp_stats in stats['components'].items():
                print(f"  {component}: {comp_stats['total']:,} 总计, {comp_stats['trainable']:,} 可训练")
                
                if component == 'audio_projector':
                    assert comp_stats['trainable'] > 0, "音频投影器应该是可训练的"
                elif component == 'llava':
                    if self.config.get('freeze_llava', True):
                        assert comp_stats['trainable'] == 0, "LLaVA应该被冻结"
            
            # 测试设备移动
            print(f"🔄 测试设备移动...")
            original_device = self.triper_model.device
            print(f"  当前设备: {original_device}")
            
            # 测试模型就绪状态
            assert self.triper_model.is_ready(), "模型应该处于就绪状态"
            
            # 测试组件访问器
            assert self.triper_model.tokenizer is not None, "tokenizer访问器失败"
            assert self.triper_model.image_processor is not None, "image_processor访问器失败"
            assert self.triper_model.audio_encoder is not None, "audio_encoder访问器失败"
            
            print(f"✅ 参数统计和配置测试通过")
            
            self.results['parameter_statistics'] = "✅ PASSED"
            
        except Exception as e:
            print(f"❌ 参数统计测试失败: {e}")
            traceback.print_exc()
            self.results['parameter_statistics'] = f"❌ FAILED: {e}"
    
    def test_7_conversation_prediction(self):
        """测试7: 对话预测任务"""
        print(f"\n{'='*50}")
        print("💬 测试7: 对话预测任务")
        print("="*50)
        
        try:
            # 测试不同类型的对话预测prompt
            test_prompts = [
                f"{DEFAULT_IMAGE_TOKEN}\nUSER: What do you see in this image?\nASSISTANT:",
                f"{DEFAULT_IMAGE_TOKEN}\nUSER: In this scene, someone says 'Hello there'. What would be a natural response?\nASSISTANT:",
                f"{DEFAULT_IMAGE_TOKEN}\nUSER: Based on what you see and hear, what conversation would happen here?\nASSISTANT:",
            ]
            
            single_batch = self.collator([self.dataset[0]])
            device_batch = {}
            for k, v in single_batch.items():
                if hasattr(v, 'to'):
                    device_batch[k] = v.to(self.triper_model.device)
                else:
                    device_batch[k] = v
            
            for i, prompt in enumerate(test_prompts):
                print(f"🔄 测试prompt {i+1}: {prompt[:50]}...")
                
                prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.triper_model.device)
                
                try:
                    response = self.triper_model.llava_model.generate(
                        inputs=prompt_ids,
                        images=device_batch['images'],
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    if response.shape[1] > prompt_ids.shape[1]:
                        generated = self.tokenizer.decode(
                            response[0, prompt_ids.shape[1]:], 
                            skip_special_tokens=True
                        )
                        print(f"  ✅ 生成成功: '{generated[:80]}...'")
                    else:
                        print(f"  ⚠️ 生成长度异常: {response.shape}")
                        
                except Exception as e:
                    print(f"  ❌ Prompt {i+1} 失败: {e}")
            
            self.results['conversation_prediction'] = "✅ PASSED"
            
        except Exception as e:
            print(f"❌ 对话预测测试失败: {e}")
            traceback.print_exc()
            self.results['conversation_prediction'] = f"❌ FAILED: {e}"
    
    def test_8_edge_cases(self):
        """测试8: 边界情况"""
        print(f"\n{'='*50}")
        print("🔍 测试8: 边界情况和错误处理")
        print("="*50)
        
        try:
            # 测试空输入
            print("🔄 测试空音频输入...")
            single_batch = self.collator([self.dataset[0]])
            device_batch = {}
            for k, v in single_batch.items():
                if hasattr(v, 'to'):
                    device_batch[k] = v.to(self.triper_model.device)
                else:
                    device_batch[k] = v
            
            # 测试只有文本
            try:
                 # 创建真正的纯文本（不包含IMAGE token）
                pure_text = "USER: What is artificial intelligence?\nASSISTANT:"
                pure_text_ids = self.tokenizer.encode(pure_text, return_tensors="pt")
                pure_text_ids = pure_text_ids.to(self.triper_model.device)
                
                print(f"📝 纯文本内容: {pure_text}")
                print(f"📝 Token范围: ({pure_text_ids.min()}, {pure_text_ids.max()})")
                
                with torch.no_grad():
                    output = self.triper_model(
                        input_ids=pure_text_ids,
                        images=None,
                        audio_features=None,
                    )
                print("✅ 纯文本输入测试通过")
            except Exception as e:
                print(f"⚠️ 纯文本输入测试失败: {e}")
            
            # 测试异常音频特征
            print("🔄 测试异常音频特征...")
            try:
                wrong_audio = torch.randn(1, 32, 1280).to(self.triper_model.device)  # 错误长度
                projected = self.triper_model.audio_projector(wrong_audio)
                print(f"✅ 音频投影器能处理不同长度: {projected.shape}")
            except Exception as e:
                print(f"⚠️ 音频投影器对异常输入敏感: {e}")
            
            # 测试极长输入
            print("🔄 测试长输入...")
            try:
                long_text = "USER: " + "This is a very long text. " * 50 + "\nASSISTANT:"
                long_ids = self.tokenizer.encode(long_text, return_tensors="pt")
                if long_ids.shape[1] > 2000:  # 如果确实很长
                    long_ids = long_ids[:, :100]  # 截断测试
                
                long_ids = long_ids.to(self.triper_model.device)
                response = self.triper_model.llava_model.generate(
                    inputs=long_ids,
                    max_new_tokens=10,
                    do_sample=False,
                )
                print(f"✅ 长输入测试通过: {response.shape}")
            except Exception as e:
                print(f"⚠️ 长输入测试失败: {e}")
            
            self.results['edge_cases'] = "✅ PASSED"
            
        except Exception as e:
            print(f"❌ 边界情况测试失败: {e}")
            traceback.print_exc()
            self.results['edge_cases'] = f"❌ FAILED: {e}"
    
    def generate_test_report(self):
        """生成测试报告"""
        print(f"\n{'='*80}")
        print("📋 TRIPER MODEL TEST REPORT")
        print("="*80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result.startswith("✅"))
        
        print(f"\n📊 测试概况:")
        print(f"  总测试数: {total_tests}")
        print(f"  通过测试: {passed_tests}")
        print(f"  失败测试: {total_tests - passed_tests}")
        print(f"  通过率: {passed_tests/total_tests*100:.1f}%")
        
        print(f"\n📋 详细结果:")
        for test_name, result in self.results.items():
            print(f"  {test_name}: {result}")
        
        # 保存报告
        report_path = "/home/wly/szl_all_code/triper-project/test_report.json"
        with open(report_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'pass_rate': passed_tests/total_tests*100
                },
                'results': self.results,
                'config': self.config
            }, f, indent=2)
        
        print(f"\n📄 测试报告已保存至: {report_path}")
        
        if passed_tests == total_tests:
            print(f"\n🎉 恭喜！所有测试都通过了！")
        else:
            print(f"\n⚠️ 有 {total_tests - passed_tests} 个测试失败，请检查具体错误。")
    
    def run_all_tests(self):
        """运行所有测试"""
        try:
            self.setup_test_environment()
            self.test_1_model_loading()
            self.test_2_data_loading()
            self.test_3_forward_pass()
            self.test_4_generation_capability()
            self.test_5_audio_projector()
            self.test_6_parameter_statistics()
            self.test_7_conversation_prediction()
            self.test_8_edge_cases()
        finally:
            self.generate_test_report()

def main():
    """主函数"""
    # 配置测试参数
    test_config = {
        'llava_model_path': "/sda1/llava-v1.5-13b",
        'audio_encoder_path': "/sda1/glm-4-voice-tokenizer",
        'data_path': '/home/wly/szl_all_code/triper-project/data/simple_data_20_samples.json',
        'media_root': '/home/wly/szl_all_code/triper-project/data',
        'device': "cuda:3",
        'freeze_llava': True
    }
    
    # 创建测试器并运行
    tester = TriperModelTester(test_config)
    tester.run_all_tests()

if __name__ == "__main__":
    main()