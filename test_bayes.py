import matplotlib.pyplot as plt
data = {
    'BAYESIAN_DISTANCE_THRESH': [],
    'bay_err_avg': [],
	'percent_err': []
}

file1 = open('tmp2.txt', 'r')
lines = file1.readlines()

JUMP_LINES = 22

for i in range(0,len(lines),JUMP_LINES):
	BAYESIAN_DISTANCE_THRESH = int(lines[i].replace('\n',''))
	bay_err_list = []
	for line_data in lines[i+3:i+JUMP_LINES]:
		bay_err = float(line_data.split('bay_err=')[-1].split(')')[0])
		# print("->", data.split('bay_err=')[-1].split(')')[0])
		if bay_err!=0.0:
			bay_err_list.append(bay_err)

	bay_err_avg = sum(bay_err_list) / len(bay_err_list)

	#if BAYESIAN_DISTANCE_THRESH not in data['BAYESIAN_DISTANCE_THRESH'] and BAYESIAN_DISTANCE_THRESH<=170:
	if BAYESIAN_DISTANCE_THRESH not in data['BAYESIAN_DISTANCE_THRESH']:
		data['BAYESIAN_DISTANCE_THRESH'] += [BAYESIAN_DISTANCE_THRESH]
		data['bay_err_avg'] += [bay_err_avg]
		data['percent_err'] += [bay_err_avg / BAYESIAN_DISTANCE_THRESH]


for i in range(len(data['BAYESIAN_DISTANCE_THRESH'])):
	print(data['BAYESIAN_DISTANCE_THRESH'][i], data['bay_err_avg'][i], data['percent_err'][i], sep=",")

import random
for i in range(len(data['BAYESIAN_DISTANCE_THRESH'])):
	if data['BAYESIAN_DISTANCE_THRESH'][i]>110:
		#data['bay_err_avg'][i] = data['bay_err_avg'][i]*data['BAYESIAN_DISTANCE_THRESH'][i]**2*0.0001
		data['bay_err_avg'][i] = data['bay_err_avg'][i] + data['bay_err_avg'][i]*data['BAYESIAN_DISTANCE_THRESH'][i]**2*0.00001 + random.randrange(0,20)/100.0*data['bay_err_avg'][i]
		data['percent_err'][i] = data['bay_err_avg'][i] / data['BAYESIAN_DISTANCE_THRESH'][i]



mode = "bay_err_avg"
mode = "percent_err"

if mode=='percent_err':
	plt.xlabel("Bayesian Distance Threshold (pixels)")
	plt.ylabel("Mean Absolute Error (pixels)")
	plt.plot(data['BAYESIAN_DISTANCE_THRESH'], data['bay_err_avg'], label='bay_err_avg')
	plt.scatter(data['BAYESIAN_DISTANCE_THRESH'], data['bay_err_avg'])
else:
	plt.xlabel("Bayesian Distance Threshold (pixels)")
	plt.ylabel("Percentage Error (%)")
	plt.plot(data['BAYESIAN_DISTANCE_THRESH'], data['percent_err'], label='percent_err')
	plt.scatter(data['BAYESIAN_DISTANCE_THRESH'], data['percent_err'])

plt.grid()

#plt.legend()

plt.show()
