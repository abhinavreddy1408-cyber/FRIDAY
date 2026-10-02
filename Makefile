.PHONY: setup run test install

setup:
	powershell -ExecutionPolicy Bypass -File ./setup.ps1

run:
	powershell -ExecutionPolicy Bypass -File ./run.ps1

test:
	powershell -ExecutionPolicy Bypass -File ./test.ps1

install: setup
