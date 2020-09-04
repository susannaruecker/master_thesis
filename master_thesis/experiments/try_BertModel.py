from transformers import BertTokenizer, DistilBertTokenizer
from master_thesis.src import models

PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model_regr = models.Bert_regression(n_outputs = 1)
model_Seq = models.Bert_sequence(n_outputs=1)

text = ["Das hier ist ein Satz, den Bert jetzt verstehen soll.", "Und noch einer", "Und dann sogar noch ein dritter, der wieder etwas länger ist."]

inputs = tokenizer(text, padding=True, return_tensors="pt", return_token_type_ids=False)
print(inputs)

print("model_regr")
output1 = model_regr(**inputs)
print(output1.size())
print(output1)

print("model_Seq")
output2 = model_Seq(**inputs)
print(output2.size())
print(output2)

print("DistilBert")
##TODO: Achtung, ich glaube, der DistilBertTokenizer hat andere special tokens
#       das Dataset schaut aber glücklicherweise in die .config des Tokenizers, also wohl kein Problem!
#       beim Collater muss das richtige padding-Symbol verwendet werden, das ist aber wohl überall 0
model_distil = models.DistilBert_sequence(n_outputs=1)
tokenizer_distil = DistilBertTokenizer.from_pretrained('distilbert-base-german-cased')
inputs_d = tokenizer_distil(text, padding = True, return_tensors='pt', return_token_type_ids=False)
print(inputs_d)
output3 = model_distil(**inputs_d)
print(output3.size())
print(output3)
