PYTHON := python

MODNAME := svmapi

all:
	$(PYTHON) setup.py build
	rm -f $(MODNAME).so
	ln -s build/lib.linux-x86_64-3.4/$(MODNAME).cpython-34.so $(MODNAME).so

install:
	$(PYTHON) setup.py install

clean:
	rm -rf build
	rm -f svmapi.so
