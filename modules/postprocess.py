import re
from typing import Dict, List, Set


class PostProcessor:
    """Post-process decoder tokens into a more natural Chinese sentence."""

    COMPLETION_VERBS: Set[str] = {
        "买", "卖", "去", "来", "看", "做", "写", "读", "吃", "喝", "打", "开", "关", "查", "查询", "提交",
        "申请", "办理", "完成", "准备", "整理", "上传", "下载", "发送", "通知", "告知", "设计", "布局", "参考",
        "说明", "修改", "学习", "帮助", "签字", "盖章", "领取", "支付", "参加", "确认", "登记", "打印",
    }

    ADJECTIVES: Set[str] = {
        "红", "红色", "大", "小", "新", "旧", "漂亮", "重要", "方便", "详细", "简单", "复杂", "高", "低",
        "长", "短", "快", "慢", "热", "冷", "好", "坏", "清楚", "正式", "临时", "残疾", "特殊", "主要",
    }

    MEASURE_WORDS: Dict[str, str] = {
        "苹果": "个", "橘子": "个", "香蕉": "根", "梨": "个", "桃子": "个", "西瓜": "个", "葡萄": "串", "草莓": "颗",
        "衣服": "件", "裤子": "条", "裙子": "条", "鞋": "双", "袜子": "双", "帽子": "顶", "包": "个", "书": "本",
        "笔": "支", "纸": "张", "桌子": "张", "椅子": "把", "电脑": "台", "手机": "部", "电视": "台", "冰箱": "台",
        "空调": "台", "车": "辆", "自行车": "辆", "公交车": "辆", "地铁": "趟", "火车": "列", "飞机": "架", "船": "艘",
        "房子": "套", "楼": "栋", "门": "扇", "窗户": "扇", "杯子": "个", "瓶子": "个", "碗": "个", "盘子": "个",
        "材料": "份", "文件": "份", "表格": "张", "申请": "份", "方案": "份", "合同": "份", "证件": "张", "卡": "张",
        "照片": "张", "老师": "位", "学生": "名", "医生": "位", "护士": "位", "工作人员": "名", "朋友": "位", "孩子": "个",
        "问题": "个", "办法": "个", "任务": "项", "项目": "个", "部门": "个", "学校": "所", "医院": "家", "公司": "家",
    }

    PUNCTUATION_MAP = {
        ",": "，",
        ".": "。",
        "!": "！",
        "?": "？",
        ";": "；",
        ":": "：",
    }

    NEGATIONS = {"不", "没", "没有"}
    NUMERALS = {"一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "两", "几", "多少"}

    def _insert_completion_marker(self, tokens: List[str]) -> List[str]:
        output: List[str] = []
        for index, token in enumerate(tokens):
            output.append(token)
            next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
            prev_token = tokens[index - 1] if index - 1 >= 0 else ""
            if token in self.COMPLETION_VERBS and prev_token not in self.NEGATIONS:
                if next_token not in {"了", "过", "着", "完", "好", "", "，", "。", "！", "？"}:
                    output.append("了")
        return output

    def _insert_measure_words(self, tokens: List[str]) -> List[str]:
        output: List[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            output.append(token)
            if index + 1 < len(tokens):
                next_token = tokens[index + 1]
                if token in self.NUMERALS or token.isdigit():
                    measure = self.MEASURE_WORDS.get(next_token)
                    if measure is not None:
                        output.append(measure)
            index += 1
        return output

    def _insert_de(self, tokens: List[str]) -> List[str]:
        output: List[str] = []
        for index, token in enumerate(tokens):
            output.append(token)
            next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
            if token in self.ADJECTIVES and next_token in self.MEASURE_WORDS:
                output.append("的")
        return output

    def _normalize_negation(self, tokens: List[str]) -> List[str]:
        output: List[str] = []
        for token in tokens:
            if token == "不" and output and output[-1] == "不":
                continue
            output.append(token)
        return output

    def _normalize_punctuation(self, text: str) -> str:
        for source, target in self.PUNCTUATION_MAP.items():
            text = text.replace(source, target)
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"([，。！？；：]){2,}", lambda match: match.group(1)[0], text)
        text = re.sub(r"([（(])\s+", r"\1", text)
        text = re.sub(r"\s+([）)])", r"\1", text)
        return text.strip("，")

    def process(self, tokens: List[str]) -> str:
        cleaned_tokens = [token.strip() for token in tokens if token and token.strip()]
        if not cleaned_tokens:
            return ""
        stage1 = self._insert_measure_words(cleaned_tokens)
        stage2 = self._insert_completion_marker(stage1)
        stage3 = self._insert_de(stage2)
        stage4 = self._normalize_negation(stage3)
        return self._normalize_punctuation("".join(stage4))
