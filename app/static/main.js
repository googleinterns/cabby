var m_topicId = '';
var stopwatch = 0;


var allQuestionsCheckboxes = [];
var commentBoxEle = null;
var questionnaireStarRatingsDone = {};

function initQuestionnaire() {
    // allQuestionsCheckboxes = [];
    // var questionnaireBody = document.getElementById('questionnaireModalBody');
    // var usedSentencesForQuestions = []; // sentences used already for empty question slots


    // // add the rating widgets for the three usability survey questions:
    // var usageQuestionTitle = document.createElement("span");
    // usageQuestionTitle.style.float = "left";
    // usageQuestionTitle.style.marginTop = "20";
    // usageQuestionTitle.style.color = "#002b33";
    // usageQuestionTitle.style.fontStyle = "italic";
    // usageQuestionTitle.style.fontWeight = "bold";
    // usageQuestionTitle.innerHTML = "Thanks! What do you think of the system?"
    // questionnaireBody.appendChild(usageQuestionTitle);
    // var usageQuestion = document.createElement("span");
    // usageQuestion.style.float = "left";
    // usageQuestion.style.marginTop = "2";
    // usageQuestion.style.marginLeft = "10";
    // usageQuestion.style.color = "#faebd7";
    // usageQuestion.innerHTML = "As a system for exploring information on a topic,"
    // questionnaireBody.appendChild(usageQuestion);
    // var questionId1 = 'r1';
    // //usageRatingEle1 = getStarRatingQuestionnaireElement(questionId1, 1, "its capabilities meet my requirments", 5, "");
    // usageRatingEle1 = getStarRatingQuestionnaireElement(questionId1, 1, "its capabilities meet the need to efficiently collect useful information for a journalistic overview", 5, "");
    // usageRatingEle1.style.float = "left";
    // usageRatingEle1.style.margin = "20 0 0 20";
    // questionnaireBody.appendChild(document.createElement("br"));
    // questionnaireBody.appendChild(usageRatingEle1);
    // questionnaireStarRatingsDone[questionId1] = false;
    // var questionId2 = 'r2';
    // usageRatingEle2 = getStarRatingQuestionnaireElement(questionId2, 2, "it is easy to use", 5, "");
    // usageRatingEle2.style.float = "left";
    // usageRatingEle2.style.margin = "-25 20 0 20";
    // questionnaireBody.appendChild(document.createElement("br"));
    // questionnaireBody.appendChild(usageRatingEle2);
    // questionnaireStarRatingsDone[questionId2] = false;
    // var responsivenessQuestion = document.createElement("span");
    // responsivenessQuestion.style.float = "left";
    // responsivenessQuestion.style.marginTop = "-7px";
    // responsivenessQuestion.style.marginLeft = "10px";
    // responsivenessQuestion.innerHTML = "During the interactive stage, how well did the responses respond to your queries?"
    // responsivenessQuestion.style.color = "#faebd7";
    // questionnaireBody.appendChild(responsivenessQuestion);
    // var questionId3 = 'r3';
    // usageRatingEle3 = getStarRatingQuestionnaireElement(questionId3, 3, "During the interactive stage, how well did the responses respond to your queries?", 5, "");
    // usageRatingEle3.style.float = "left";
    // usageRatingEle3.style.margin = "-30px 50px 0px 212px";
    // questionnaireBody.appendChild(document.createElement("br"));
    // questionnaireBody.appendChild(usageRatingEle3);
    // questionnaireStarRatingsDone[questionId3] = false;
    
    // questionnaireBody.appendChild(document.createElement("br"));

    // // add a comments box for the user to write any comments:
    // commentBoxEle = document.createElement("textarea");
    // commentBoxEle.placeholder = "comments (optional)";
    // commentBoxEle.rows = 3;
    // //commentBoxEle.cols = 50;
    // commentBoxEle.style.width = "300px";
    // commentBoxEle.style.marginTop = "10px";
    // questionnaireBody.appendChild(commentBoxEle);
    
    // add a submit button at the bottom of the questionnaire:
    var submitButtonEle = document.createElement("div");
    submitButtonEle.classList.add("submitButton");
    // questionnaireBody.appendChild(submitButtonEle);

    submitButtonEle.addEventListener("click", submitQuestionnaire);
    submitButtonEle.appendChild(document.createTextNode("Submit"));

}

// function mturk_submit()
// {
//     const form = document.getElementById("mturk_form");
//     const turk_submit = getUrlParam('turkSubmitTo');
//     if (turk_submit) {
//         form.action=turk_submit + '/mturk/externalSubmit';
//         document.getElementById("assignmentId").value = getUrlParam('assignmentId');
//     }
//     form.submit();

// }


// function setAMTvalues(assignmentIdVal, hitIdVal, workerIdVal, turkSubmitToVal) {
//     var givenTurkSubmitTo;
//     if (assignmentIdVal != null) {
//         assignmentId = assignmentIdVal;
//     }
//     else {
//         assignmentId = '';
//     }
//     if (hitIdVal != null) {
//         hitId = hitIdVal;
//     }
//     else {
//         hitId = '';
//     }
//     if (workerIdVal != null) {
//         workerId = workerIdVal;
//     }
//     else {
//         workerId = '';
//     }
//     if (turkSubmitToVal != null) {
//         givenTurkSubmitTo = turkSubmitToVal.replace('%3A%2F%2F', '://'); // replace hexa "://" if given
//     }
//     else {
//         givenTurkSubmitTo = '';
//     }
    
    
//     document.getElementById('assignmentId').value = assignmentId;
//     document.getElementById('hitId').value = hitId;
//     document.getElementById('workerId').value = workerId;
    
//     var turkSubmitToturkSubmitTo = '';
//     if (givenTurkSubmitTo == 'https://workersandbox.mturk.com' || givenTurkSubmitTo == 'https://workersandbox.mturk.com/')
//         turkSubmitTo = 'https://workersandbox.mturk.com/mturk/externalSubmit';
//     else if (givenTurkSubmitTo == 'https://www.mturk.com' || givenTurkSubmitTo == 'https://www.mturk.com/')
//         turkSubmitTo = 'https://www.mturk.com/mturk/externalSubmit';
//     else
//         turkSubmitTo = givenTurkSubmitTo + 'externalSubmit/index.html';
//     document.getElementById('turkSubmitTo').value = turkSubmitTo;
    
//     document.getElementById('turkSubmit').action = turkSubmitTo;
// }

function practiceTaskMessage(messageHtml, functionToExecute, isOkCancel=false) {
    if (isPracticeTask) {
        (function () {
            document.getElementById("mainDiv").style.filter = "blur(4px)";
            document.getElementById("pageCover").style.display = "block";
            var dialogDiv = document.getElementById("practiceDirectionsMessageDialog");
            var okButton = document.getElementById("practiceDirectionsMessageDialogOkButton");
            var cancelButton = document.getElementById("practiceDirectionsMessageDialogCancelButton");
            var messageArea = document.getElementById("practiceDirectionsMessageDialogMessageArea");

            okButton.addEventListener("click", function() {
                closePracticeDirectionsMessageDialog();
                functionToExecute();
            });
            cancelButton.addEventListener("click", function() {
                closePracticeDirectionsMessageDialog();
            });
            cancelButton.style.display = isOkCancel ? "block" : "none";
            okButton.style.marginLeft = isOkCancel ? "calc(50% - 125px)" : "calc(50% - 60px)"
            messageArea.innerHTML = messageHtml;
            dialogDiv.style.display = "block";
            window.scrollTo(0, 0);
        })();
    }
    else {
        functionToExecute();
    }
}

function submitQuestionnaire() {
    //if (needIterationStarRating && !lastIterationRated) {

    practiceTaskMessage("Thank you!<br>Once you've finished the two practice tasks, we will check your work.<br>If you qualify, there will be <b>more of these tasks</b>, <u>without the guidance messages</u>.<br><br>Your sincere work can assist us in understanding how to build better systems for knowledge acquisition. <span style='font-size:30px;'>&#x1F680;</span><br><br><b>Thanks for your help!</b> <span style='font-size:30px;'>&#x1F60A;", function() {
        allAnswers = {};
        for (var i = 0; i < allQuestionsCheckboxes.length; i++) {
            allAnswers[allQuestionsCheckboxes[i].value] = allQuestionsCheckboxes[i].checked;
        }
        sendRequest({"clientId": clientId, "request_submit": {"answers": allAnswers, 'timeUsed': stopwatch, 'comments': commentBoxEle.value}});
        // response goes to submitFinal in general.js
    });
    
}



function sendRequest(jsonStr) {
    // Sending and receiving data in JSON format using POST method
    var xhr = new XMLHttpRequest();
    var url = requestUrl;
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            try {
                var jsonObj = JSON.parse(xhr.responseText);
                handleJsonReply(jsonObj);
            } catch (e) {
                //alert(xhr.responseText);
                alert("Error getting response: " + e);
            }
        }
        else if (xhr.readyState === 4 && xhr.status === 503) {
            alert("Service seems to be down.");
            // The web service is down on the internal server!
        }
    };
    var data = JSON.stringify(jsonStr);
    xhr.send(data);
    isWaitingForResponse = true;
}

function handleJsonReply(jsonObj) {
    isWaitingForResponse = false;
    if ('error' in jsonObj) {
        if (curLoadingInicatorElement != null) {
            exploreList.removeChild(curLoadingInicatorElement);//exploreList.lastChild);
            curLoadingInicatorElement = null;
        }
        if (isWaitingForInitial) {
            isWaitingForInitial = false;
            setNoTopicChosen();
        }
        alert("Error: " + jsonObj["error"]);
    }
    else if ("reply_get_topics" in jsonObj) {
        setTopicsList(jsonObj["reply_get_topics"]["topicsList"]);
    }
    else if ("reply_get_initial_summary" in jsonObj) {
        setTopic(jsonObj["reply_get_initial_summary"])
    }
    else if ("reply_set_start" in jsonObj) {
        // nothing to do
    }
    else if ("reply_query" in jsonObj) {
        setQueryResponse(jsonObj["reply_query"])
    }
    else if ("reply_set_question_answer" in jsonObj) {
        // nothing to do
    }
    else if ("reply_submit" in jsonObj) {
        submitFinal(jsonObj["reply_submit"]["success"]);
    }
    else if ("reply_set_iteration_rating" in jsonObj) {
        // nothing to do
    }
    else if ("reply_set_questionnaire_rating" in jsonObj) {
        // nothing to do
    }
    else {
        if (curLoadingInicatorElement != null) {
            exploreList.removeChild(curLoadingInicatorElement);//exploreList.lastChild);
            curLoadingInicatorElement = null;
        }
        if (isWaitingForInitial) {
            isWaitingForInitial = false;
            setNoTopicChosen();
        }
        alert("Error: No relevant response recieved from server.");
    }
}



function submitFinal() {
    SubmitToAMT(function(success) {
        if (success) {
            // document.getElementById('finishMessage').innerHTML = "Session submitted successfully.";
            alert("Fake sent to AMT submit successfully.")
        }
        else {
            //document.getElementById('finishMessage').innerHTML = "Failed to submit session.";
            alert("Fake failed to send to AMT submit.")
        }
    });
}

function SubmitToAMT(baseUrl, assignmentId) {
    // TODO: update the CGI paramaeters of the submit URL
    var baseUrl = document.getElementById('turkSubmit').action;
    var assignmentId = document.getElementById('assignmentId').value;

    

    var fullUrl = baseUrl + '?assignmentId=' + assignmentId;
    // fullUrl += '&clientId=' + clientId;
    // fullUrl += '&topicId=' + m_topicId;
    // fullUrl += '&timeAllowed=' + timeAllowed;
    // fullUrl += '&totalTextLength=' + totalTextLength;
    // fullUrl += '&timeUsed=' + stopwatch;
    // fullUrl += '&questionnaireBatchInd=' + questionnaireBatchInd;
    // fullUrl += '&comments=' + commentBoxEle.value.replace(/(\r\n|\n|\r)/gm," ");

    
    // // questionnaire answers:
    // for (var i = 0; i < allQuestionsCheckboxes.length; i++) {
    //     fullUrl += ('&' + allQuestionsCheckboxes[i].value + '=' + allQuestionsCheckboxes[i].checked);
    // }
    // update the url with the parameters

    alert(fullUrl);

    document.getElementById('turkSubmit').action = fullUrl;
    document.getElementById("turkSubmit").submit(); // TODO: comment/uncomment
    // callbackSuccess(true);
}
