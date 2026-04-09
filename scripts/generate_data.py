from data_generator import generate_data

specs = [
# base environments
{
    "environment": "flatland",
    "name": "cyclic",
    "specs" : {
        "color_type": "cyclic",
        "n_pos": 5,
    },
},
{
    "environment": "flatland",
    "name": "permutation",
    "specs" : {
        "n_pos": 5,
        "color_type": "permutation",
    },
},
{
    "environment": "coil",
    "name": "2",
    "specs" : {
        "objects": [74,85],
        "rotations": [7,5],
    },
},
{
    "environment": "coil",
    "name": "3",
    "specs" : {
        "objects": [74,85,21],
        "rotations": [5,5,3],
    },
},
{
    "environment": "shapes",
    "name": "ss2",
    "specs" : {
        "subsampling": 2,
    },
},
# experiments for entangled actions
{
    "environment": "coil",
    "name": "entangled1",
    "specs" : {
        "objects": [74,85],
        "rotations": [7,5],
        "entangled_actions": 1
    },
},
{
    "environment": "coil",
    "name": "entangled2",
    "specs" : {
        "objects": [74,85],
        "rotations": [7,5],
        "entangled_actions": 2
    },
},
{
    "environment": "coil",
    "name": "entangled3",
    "specs" : {
        "objects": [74,85],
        "rotations": [7,5],
        "entangled_actions": 3
    },
},
# experiments for random available actions
{
    "environment": "coil",
    "name": "randomaction0",
    "specs" : {
        "objects": [74,85,21],
        "rotations": [7,5,3],
        "available_rotations": [[1,3], [3,4], [1]],
        "available_permut" : [1],
        "e": True
    },
},
{
    "environment": "coil",
    "name": "randomaction1",
    "specs" : {
        "objects": [74,85,21],
        "rotations": [7,3,5],
        "available_rotations": [[2,6], [2], [1]],
        "available_permut" : [5],
        "e": True
    },
},
{
    "environment": "coil",
    "name": "randomaction2",
    "specs" : {
        "objects": [74,85,21,8],
        "rotations": [7,5,3,3],
        "available_rotations": [[1,3], [3,4], [1], [2]],
        "available_permut" : [17,22],
        "e": False
    },
},
# experiment for Lie group with mpi3d
{
    "environment": "mpi3d",
    "name": "lie",
    "specs" : {
    },
},
]

for spec in specs :
    generate_data(**spec,
                  verbose = True)
