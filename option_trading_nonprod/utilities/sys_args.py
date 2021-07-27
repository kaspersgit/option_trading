import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", required=False)
parser.add_argument("-d", "--date", required=False)
args = parser.parse_args()
print(f'Hi {args.name}, Welcome to {args.date}')