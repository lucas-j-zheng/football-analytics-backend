.PHONY: train serve loadtest report

train:
	python -c "from decision_service.train import train_models; print(train_models())"

serve:
	python run_decision_service.py

loadtest:
	locust -f ../scripts/locust_decision.py --headless -u 1000 -r 200 -t 1m

report:
	python -c "print('Generate PDF case study placeholder')"


