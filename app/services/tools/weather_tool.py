"""
天气查询工具：模拟调用天气 API
"""

from langchain_core.tools import tool


@tool
def get_current_weather(city: str) -> str:
    """
    获取指定城市的当前天气信息。

    参数:
        city: 城市名称，例如"上海"、"北京"、"深圳"

    返回:
        天气信息字符串
    """
    # 模拟调用天气 API
    weather_data = {
        "上海": "25度，晴朗，空气质量良好",
        "北京": "18度，多云，有轻度雾霾",
        "深圳": "28度，阵雨，湿度较大",
    }

    result = weather_data.get(city, f"{city}的天气数据暂时无法获取")
    print(f"🌤️  [Weather Tool] get_current_weather(city='{city}') -> {result}")
    return result
