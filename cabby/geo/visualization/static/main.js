function mturk_submit()
{
    const form = document.getElementById("mturk_form");
    const turk_submit = getUrlParam('turkSubmitTo');
    if (turk_submit) {
        form.action=turk_submit + '/mturk/externalSubmit';
        document.getElementById("assignmentId").value = getUrlParam('assignmentId');
    }
    form.submit();

}