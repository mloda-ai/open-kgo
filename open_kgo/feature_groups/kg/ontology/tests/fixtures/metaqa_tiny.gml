graph [
  directed 1
  node [
    id 0
    label "The Dark Knight"
    type "Movie"
  ]
  node [
    id 1
    label "Inception"
    type "Movie"
  ]
  node [
    id 2
    label "The Godfather"
    type "Movie"
  ]
  node [
    id 3
    label "Christopher Nolan"
    type "Person"
  ]
  node [
    id 4
    label "Francis Ford Coppola"
    type "Person"
  ]
  node [
    id 5
    label "Christian Bale"
    type "Person"
  ]
  node [
    id 6
    label "Leonardo DiCaprio"
    type "Person"
  ]
  node [
    id 7
    label "Action"
    type "Genre"
  ]
  node [
    id 8
    label "Crime"
    type "Genre"
  ]
  node [
    id 9
    label "Drama"
    type "Genre"
  ]
  edge [
    source 0
    target 3
    relation "directed_by"
  ]
  edge [
    source 0
    target 5
    relation "starred_actors"
  ]
  edge [
    source 0
    target 7
    relation "has_genre"
  ]
  edge [
    source 0
    target 8
    relation "has_genre"
  ]
  edge [
    source 1
    target 3
    relation "directed_by"
  ]
  edge [
    source 1
    target 6
    relation "starred_actors"
  ]
  edge [
    source 1
    target 7
    relation "has_genre"
  ]
  edge [
    source 2
    target 4
    relation "directed_by"
  ]
  edge [
    source 2
    target 8
    relation "has_genre"
  ]
  edge [
    source 2
    target 9
    relation "has_genre"
  ]
  edge [
    source 8
    target 3
    relation "directed_by"
  ]
]
