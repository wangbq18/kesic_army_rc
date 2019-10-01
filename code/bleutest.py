from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'the', 'test'], ['this', 'is' , 'my','test']]
candidate = ['this', 'is', 'what', 'test']
score = sentence_bleu(reference, candidate)
print(score)