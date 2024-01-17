import argparse

# 创建第一个解析器对象
parser1 = argparse.ArgumentParser(add_help=False)
parser1.add_argument('--a', help='a help')

# 创建第二个解析器对象
parser2 = argparse.ArgumentParser(add_help=False)
parser2.add_argument('--b', help='b help')

# 创建第三个解析器对象
parser3 = argparse.ArgumentParser(add_help=False)
parser3.add_argument('--c', help='c help')

# 创建一个包含所有要合并的解析器对象的列表
parsers = [parser1, parser2, parser3]

# 创建一个新的解析器，并将所有要合并的解析器作为父解析器
merged_parser = argparse.ArgumentParser(parents=parsers)

# 解析命令行参数
args = merged_parser.parse_args()

# 打印解析结果
print(args.a)
print(args.b)
print(args.c)
