from typing import List, Union, Dict
from happy_hub.functions import OpenAIFunction


def fun_chat():
    r"""được sử dụng để trả lời các câu hỏi của người dùng mang tính trò chuyện, 
    không có nội dung cụ thể liên quan đến MB.
    """
    pass

def mb_info_query(query: str):
    r"""Hàm này được sử dụng để trả lời khi người dùng hỏi thông tin liên quan đến ngân hàng MB như:
    - Các thông tin chung về MB.
    - Các quy trình thủ tục cho nhân sự mới ở MB.
    - Những tiện ích khác của MB như là gym, chỗ ăn uống, đãi ngộ...
    - MB có cái gì hay ho.

    Args:
        query (str): câu truy vấn ngắn gọn đủ ý nhất để lấy thông tin.
    
    """
    pass


RETRIEVAL_FUNCS: List[OpenAIFunction] = [
    OpenAIFunction(func)
    for func in [
        fun_chat,
        mb_info_query
    ]
]


if __name__=="__main__":
    import json
    
    for func in RETRIEVAL_FUNCS:
        print(json.dumps(func.get_openai_tool_schema(), indent=4))