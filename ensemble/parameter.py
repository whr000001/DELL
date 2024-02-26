dataset_names = {
    'TASK1': ['Pheme', 'LLM-misinformation'],
    'TASK2': ['MFC', 'semeval-2023'],
    'TASK3': ['generated', 'semeval-2020', 'semeval-2023']
}
TASK2_label = {
    'MFC': ['Morality', 'Policy Prescription and Evaluation', 'Crime and Punishment', 'Health and Safety',
            'Security and Defense', 'Economic', 'Political', 'Public Sentiment', 'Cultural Identity',
            'Capacity and Resources primany', 'Quality of life', 'Legality, Constitutionality, Jurisdiction',
            'Policy Presecription and Evaluation', 'Capacity and Resources',
            'Fairness and Equality', 'External regulation and reputation', 'Quality of Life',
            'External Regulation and Reputation'],
    'semeval-2023': ['Security_and_defense', 'Fairness_and_equality', 'Political', 'Capacity_and_resources',
                     'Economic', 'Morality', 'Policy_prescription_and_evaluation',
                     'Legality_Constitutionality_and_jurisprudence', 'External_regulation_and_reputation',
                     'Quality_of_life', 'Health_and_safety', 'Cultural_identity', 'Crime_and_punishment',
                     'Public_opinion']
}
TASK3_label = {
    'generated': ['flag waving', 'straw man', 'false dilemma', 'name calling', 'guilt by association',
                  'whataboutism', 'repetition', 'obfuscation', 'slogans', 'conversation killer', 'appeal to hypocrisy',
                  'doubt', 'loaded language', 'exaggeration', 'causal oversimplification', 'appeal to fear',
                  'appeal to popularity', 'red herring', 'appeal to authority'],
    'semeval-2020': ['Appeal_to_fear-prejudice', 'Black-and-White_Fallacy', 'Name_Calling-Labeling', 'Slogans',
                     'Whataboutism-Straw_Men-Red_Herring', 'Exaggeration-Minimisation', 'Loaded_Language',
                     'Repetition', 'Causal_Oversimplification', 'Bandwagon-Reductio_ad_hitlerum', 'Flag-Waving',
                     'Thought-terminating_Cliches', 'Appeal_to_Authority', 'Doubt'],
    'semeval-2023': ['Conversation_Killer', 'False_Dilemma-No_Choice', 'Appeal_to_Popularity', 'Doubt', 'Flag_Waving',
                     'Slogans', 'Whataboutism', 'Straw_Man', 'Loaded_Language', 'Name_Calling-Labeling',
                     'Obfuscation-Vagueness-Confusion', 'Appeal_to_Fear-Prejudice', 'Causal_Oversimplification',
                     'Red_Herring', 'Repetition', 'Exaggeration-Minimisation', 'Appeal_to_Authority',
                     'Guilt_by_Association', 'Appeal_to_Hypocrisy']
}


class TAPEDescription:
    def __init__(self):
        data = {
            'TASK1': {},
            'TASK2': {},
            'TASK3': {}
        }
        prompt = 'Question: Please determine whether the following news is real or fake, and provide your reasoning.\n'
        data['TASK1']['LLM-misinformation'] = prompt
        data['TASK1']['Pheme'] = prompt
        for dataset in dataset_names['TASK2']:
            prompt = 'Task:\nFraming is a strategic device and a central concept in political communication for ' \
                         'representing different salient aspects and ' \
                         'perspectives to convey the latent meaning of an issue. '
            prompt += 'Which framings does the news contain? Please choose from: '
            for item in TASK2_label[dataset]:
                prompt += '\'{}\''.format(item) + ', '
            prompt = prompt[:-2] + '. Provide your reasoning.\n'
            data['TASK2'][dataset] = prompt
        for dataset in dataset_names['TASK3']:
            prompt = 'Question:\nPropaganda techniques are methods used in propaganda to convince an audience to ' \
                     'believe what the propagandist wants them to believe. '
            prompt += 'Which propaganda techniques does the news contain? Please choose from: '
            for item in TASK3_label[dataset]:
                prompt += '\'{}\''.format(item) + ', '
            prompt = prompt[:-2] + '. Provide your reasoning.\n'
            data['TASK3'][dataset] = prompt
        self.data = data


class TaskDescription:
    def __init__(self):
        data = {
            'TASK1': {},
            'TASK2': {},
            'TASK3': {}
        }
        prompt = 'Task:\nPlease determine whether the news is real or fake.\n'
        data['TASK1']['LLM-misinformation'] = prompt
        data['TASK1']['Pheme'] = prompt
        for dataset in dataset_names['TASK2']:
            prompt = 'Task:\nFraming is a strategic device and a central concept in political communication for ' \
                         'representing different salient aspects and ' \
                         'perspectives to convey the latent meaning of an issue. '
            prompt += 'Which framings does the news contain? Please choose from: '
            for item in TASK2_label[dataset]:
                prompt += '\'{}\''.format(item) + ', '
            prompt = prompt[:-2] + '.\n'
            data['TASK2'][dataset] = prompt
        for dataset in dataset_names['TASK3']:
            prompt = 'Task:\nPropaganda techniques are methods used in propaganda to convince an audience to ' \
                     'believe what the propagandist wants them to believe. '
            prompt += 'Which propaganda techniques does the news contain? Please choose from: '
            for item in TASK3_label[dataset]:
                prompt += '\'{}\''.format(item) + ', '
            prompt = prompt[:-2] + '.\n'
            data['TASK3'][dataset] = prompt
        self.data = data



