def load_and_clean_data(filepath):
    questions = []
    answers = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                questions.append(parts[0])
                answers.append(parts[1])
    return questions, answers
