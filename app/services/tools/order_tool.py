"""
订单查询工具：模拟调用订单服务
"""

from langchain_core.tools import tool


@tool
def query_database_order(order_id: str) -> str:
    """
    查询订单的当前状态。

    参数:
        order_id: 订单编号，例如"9982"、"ORD123456"

    返回:
        订单状态信息
    """
    # 模拟调用订单服务
    print(f"📦 [Order Tool] query_database_order(order_id='{order_id}')")
    return f"订单 {order_id} 正在配送中，预计明天下午送达"
