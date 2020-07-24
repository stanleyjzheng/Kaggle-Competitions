## Open Source Imaging Consortium (OSIC) Pulmonary Fibrosis Progression

##### Machine learning solution using regression

##### Placed #28 out of 279 teams (top 10% of all competitors)

A few notable changes in the model that have affected performance (note that a numbers lcoser to 0 are more desirable):

- Version 1: Scored -6.8781
- Version 2: Scored -6.8354 (Changed decay from 0.05 to 0.01)
- Version 3: Scored -6.8717 (Changed 2 dense layers "d1" and "d2" from 100 to 110 layers)
- Version 4: Scored -6.8464 (Changed 2 dense layers "d1" and "d2" back to 100 layers and increased epochs to 900 from 800)
- Version 5: Scored -6.8385 (Decreased epochs from 900 to 700)
- Version 6: Scored -6.8447 (Increased epochs to 850 from 700)
- Version 7: Scored -6.8463 (Decreased epochs to 810 from 850)
- Version 8: Scored -6.8391 (Decreased epochs from 810 to 790)
- Version 9: Scored -6.9176 (Add third dense layer of 100 neurons "d3" and add dropout of 0.1, increase epochs from 790 to 1000)
- Version 10: Scored -6.9026 (Removed third dense layer of 100 neurons "d3" and decrease dropout from 0.1 to 0.05)
- Version 11: Scored -6.8475 (Removed dropout and add sigmoid in place of linear output "p1")
- Version 12: Scored -6.8476 (Increase epochs from 1000 to 1500)
- Version 13: Scored -6.8533 (Decrease decay from 0.01 to 0.005 and decrease epochs from 1500 to 600)
- Version 14: Scored  -6.8790 (Replace sigmoid "p1" activation with relu, increase epochs from 600 to 800)


![](https://img.techpowerup.org/200717/screenshot-20200717-155422.jpg)
