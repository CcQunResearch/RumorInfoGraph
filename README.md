# 实验记录

1. Weibo

实验结果：

| id | mode | label limit num | unlabel limit num | clean | unlabel post num | word embedding size | graph embedding size | batch size | convergence epoch | val acc | test acc | test prec(T/F) | test rec(T/F) | f1(T/F) | 
| :----:| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 1 | semi-sup | 600 | no limit | 清理“转发微博” | 60198 | 100 | 64 | **8** | 32 | 0.955 | 0.934 | 0.920/0.947 | 0.941/0.927 | 0.930/0.937 |
| 2 | semi-sup | 600 | no limit | 清理“转发微博” | 60198 | 100 | 64 | **12** | 48 | 0.948 | 0.923 | 0.886/0.961 | 0.959/0.891 | 0.921/0.925 |
| 3 | semi-sup | 600 | no limit | 清理“转发微博” | 60198 | 100 | 64 | **16** | 36 | 0.951 | 0.925 | 0.897/0.953 | 0.950/0.903 | 0.922/0.928 |
| 4 | semi-sup | 600 | no limit | **清理“转发微博”，清理“回复@”** | 60198 | 100 | 64 | 8 | 68 | 0.955 | 0.936 | 0.936/0.935 | 0.928/0.943 | 0.932/0.939 |
| 5 | semi-sup | 600 | no limit | **清理“转发微博”，清理“回复@”** | 60198 | 100 | 64 | 8 | 70 | 0.957 | 0.936 | 0.957/0.919 | 0.905/0.963 | 0.930/0.940 |

word embedding参数：

| id | algo | word embedding size |train w2v post num | window | min count | epoch |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 1 | skip-gram | 100 | all label + 60198 unlabel | 5 | 5 | 10 |
| 2 | skip-gram | 100 | all label + 60198 unlabel | 5 | 5 | 10 |
| 3 | skip-gram | 100 | all label + 60198 unlabel | 5 | 5 | 10 |
| 4 | skip-gram | 100 | all label + 60198 unlabel | 5 | 5 | 10 |
| 5 | skip-gram | 100 | all label + 60198 unlabel | 5 | 5 | 10 |

2. Tweet

Waiting....

# 备注

1. 数据集id错误的修改

把微博数据集```3495745049431351.json```的源帖子```mid```改成了```3495745049431350```。

2. 微博标注数据集

处理后的微博标注数据集一般满足评论的id大于其父节点的id，但是有以下几个不满足帖子不满足：

| filename | cid | pid |
| :----:| :----: | :----: |
| 3521098215849074.json | 6429 | 6430 |
| 3567091888659283.json | 1169 | 1172 |
| 3515863422717161.json | 12325 | 12328 |
| 3515863422717161.json | 12621 | 12624 |
| 3512941028164269.json | 4781 | 4782 |

cid: comment id

pid: parent id

