result = []

with open("students.txt", "r") as f:
    for line in f.readlines():
        name, *scores = line.split()
        mid, fin, assign = map(int, scores)
        weighted_avg = mid*0.4 + fin*0.4 + assign*0.2

        grade = ""
        if weighted_avg >= 90:
            grade = "A"
        elif weighted_avg >= 70:
            grade = "B"
        elif weighted_avg >= 40:
            grade = "C"
        else:
            grade = "D"

        result.append(f"{name} {grade}\n")

with open("202401833.txt", "w") as f:
    f.writelines(result)


