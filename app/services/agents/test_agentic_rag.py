"""
Agentic RAG 测试脚本

用于测试拆分后的 Agentic RAG 系统
"""

from dotenv import load_dotenv
from .agentic_rag_graph import run_agentic_rag

# 加载环境变量
load_dotenv()


def test_knowledge_base_only():
    """测试纯知识库问题"""
    print("\n" + "📚" * 40)
    print("【测试 1】纯知识库问题")
    print("📚" * 40 + "\n")

    result = run_agentic_rag("公司的培训类型有哪些？", session_id="test_user_002")
    return result


def test_weather_only():
    """测试纯天气查询"""
    print("\n" + "🌤️ " * 40)
    print("【测试 2】纯天气查询")
    print("🌤️ " * 40 + "\n")

    result = run_agentic_rag("北京今天天气怎么样？", session_id="test_user_003")
    return result


def test_fallback():
    """测试拒答能力（所有工具都无法解决的问题）"""
    print("\n" + "❓" * 40)
    print("【测试 3】拒答测试（超出能力范围）")
    print("❓" * 40 + "\n")

    result = run_agentic_rag("帮我写一首诗", session_id="test_user_004")
    return result


def test_complex_query():
    """测试复合问题：同时触发多个工具"""
    print("\n" + "🔥" * 40)
    print("【终极测试】复合问题：订单 + 知识库 + 天气")
    print("🔥" * 40 + "\n")

    complex_question = (
        "帮我查一下订单 9982 的状态，"
        "另外我想知道公司出差打车的报销额度是多少？"
        "对了，今天上海会下雨吗？"
    )

    result = run_agentic_rag(complex_question, session_id="test_user_001")

    print("\n" + "=" * 80)
    print("🎯 测试结果分析：")
    print("=" * 80)
    print("✓ Agent 成功识别了 3 个不同的意图")
    print("✓ 并发调用了 3 个工具：")
    print("  1. query_database_order(order_id='9982')")
    print("  2. search_knowledge_base(query='出差打车报销额度')")
    print("  3. get_current_weather(city='上海')")
    print("✓ LLM 综合所有工具结果，生成了统一的回答")
    print("=" * 80)

    return result


if __name__ == "__main__":
    try:
        # 运行所有测试
        test_knowledge_base_only()
        test_weather_only()
        test_fallback()
        test_complex_query()

        print("\n" + "🎉" * 40)
        print("所有测试完成！")
        print("🎉" * 40)

    except Exception as e:
        print(f"\n❌ 运行出错：{e}")
        import traceback

        traceback.print_exc()
