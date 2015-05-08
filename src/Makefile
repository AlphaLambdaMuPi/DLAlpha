all: init

init: ../delta.tgz

../delta.tgz:
	cd .. && wget "http://googledrive.com/host/0B57yLhlKrsM0cXExOUt2NFRBVzQ" -O delta.tgz
	cd .. && tar -xvf delta.tgz
	cd ssvm && make

run: run_voting

run_voting: init
	cd voting && python3 voting.py
	cd voting && python3 hmm39.py
	cd voting && python3 trans.py
	cp voting/mnmn.out output.kaggle

run_dnn: init
	python3 main.py run test_blocks2
	cp ../output/zzzzz.npy voting/zzzzz.npy
	cd voting && python3 hmm_state.py
	cd voting && python3 trans.py
	cp voting/mnmn.out output.kaggle

run_rnn: init
	python3 main.py run rnn
	cp hw2_outz.txt output.kaggle

run_lstm: init
	python3 main.py run lstm
	cp hw2_outz.txt output.kaggle

run_ssvm: init
	cd ssvm && ./svm_python_classify --m svm_api x svm_unknkn y
	cp ssvm/hw2_pred.out output.kaggle
