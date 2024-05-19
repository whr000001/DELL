import json

tasks = {
    'TASK1': ['Pheme', 'LLM-mis'],
    'TASK2': ['MFC', 'SemEval-23F'],
    'TASK3': ['Generated', 'SemEval-20', 'SemEval-23P']
}  # Our employed dataset in the paper. If you want to adopt your data, we will upload the complete code before June


def main():
    # example code for reading the original dataset and networks
    for task in tasks:
        for dataset in tasks[task]:
            data = json.load(open(f'datasets/{task}/{dataset}.json'))
            network_dir = f'networks/{task}_{dataset}'
            for index, item in enumerate(data):
                print(item[0])  # the news content
                print(item[1])  # the label of the news article; TASK1: 1 for fake, 0 for real
                network = json.load(open(f'{network_dir}/{index}.json'))  # the network of the corresponding news article
                print(network.keys())
                print(network['father'])  # the father of each comment node, None means this node is the root node (the news article)
                print(network['users'])  # the user profile prompt
                print(network['res'])  # the comment content, index 0 is the news article
                print(network['reasons'])  # the reason why LLMs select to comment on the father node (not used in our paper)
                input()


if __name__ == '__main__':
    main()
