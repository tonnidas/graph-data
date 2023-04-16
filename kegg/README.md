# KEGG Dataset

## Download and parse data

- Parser repository: https://github.com/everest-castaneda/knext

- Download data

```
$ knext get-kgml hsa
```

- Parse data

```
$ parse-genes folder kgml_hsa
$ parse-genes folder kgml_hsa --unique
```

## Process raw data

```
$ python process.py
```

## Datasets

| Dataset         | Nodes | Edges  |
|-----------------|-------|--------|
| hsa04740        | 434   | 83516  |
| hsa04740_unique | 844   | 164131 |
| hsa05168        | 479   | 56490  |
| hsa05168_unique | 518   | 56953  |
| hsa04151        | 277   | 4675   |
| hsa04151_unique | 281   | 4675   |
| hsa04010        | 301   | 4359   |
| hsa04010_unique | 316   | 4393   |
| hsa04014        | 233   | 3495   |
| hsa04014_unique | 273   | 3625   |
