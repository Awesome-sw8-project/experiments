import evaluation.evaluator

# Writes evaluation output.
def write(evaluator, output_name):
    with open(output_name, "w") as file:
        file.write("MPE:\t\t" + str(evaluator.get_mpe()))
        file.write("RMSE:\t\t" + str(evaluator.get_rmse()))
        file.write("\n\nIndex\t\tPositioning error\n")
        __write_pes(evaluator, file)

# Writes array of positioning errors.
def __write_pes(evaluator, file):
    results = evaluator.get_pe()

    for i in range(len(results)):
        file.write(str(i) + "\t\t\t" + str(results[i]) + "\n")

# Writes integer value as MPE average.
def write_mpe_avg(path, val):
    with open(path + "average.txt", "w") as file:
        file.write("MPE average\n")
        file.write(str(val))
