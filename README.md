本论文题目为基于深度强化学习的德州扑克AI算法优化
结果储存在result.xlsx，以每个图的数据进行呈现，包括中期报告和论文的数据

本论文三个实验环境为：
    Limit leduc holdem poker(有限注德扑简化版):
        文件夹为limit_leduc，写代码的时候为了简化，使用的环境命名为NolimitLeducholdemEnv，但实际上是limitLeducholdemEnv

    Nolimit leduc holdem poker(无限注德扑简化版):
        文件夹为nolimit_leduc_holdem3，使用环境为NolimitLeducholdemEnv（chips=10）
        
    Limit holdem poker(有限注德扑)
        文件夹为limitholdem，使用环境为LimitholdemEnv

本论文所设计的agent位于"/实验环境/agents/DeepCFRagent3.py"，是由DeepCFRagent改进来的agent，在实验中，我们与CFR，CFR+，MCCFR，DeepCFR进行对比，Limit leduc holdem poker和Nolimit leduc holdem poker使用exploitability进行评估（exploitability衡量算法与纳什均衡的距离），Limit holdem poker环境过大，使用与RandomAgent作战的reward作为评估指标

本论文工作量量：
    1.本论文所使用的agent，800+行
    2.本论文复现出的CFR，CFR+，MCCFR，DeepCFR算法，因为CFR，CFR+，MCCFR重复部分较高，每个算法400行左右，DeepCFR为600行，以上算法都未开源
    3.本文使用的环境，我们使用RLcard作为我们的底层，每个环境大约为500行左右，重复部分较高

本文为online-learning，无数据集
