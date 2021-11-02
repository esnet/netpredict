const IncomingWebhook = require('@slack/webhook').IncomingWebhook;
const url = "https://hooks.slack.com/services/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";

const webhook = new IncomingWebhook(url);

messagebody="Take a coffee break."

// Send the notification - Gets callled by Cloud Scheduler
module.exports.sendToSlack = () => {
  (async (msg) => {
    await webhook.send({
      icon_emoji: ':male-police-officer:',
      text: `@here ${JSON.stringify(msg)}`
    });
  })();
};