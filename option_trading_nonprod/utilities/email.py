# Send an HTML email with an embedded image and a plain text message for
# email clients that don't want to display the HTML.

from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import io
import smtplib

def df_to_bytes(df):
	with io.StringIO() as buffer:
		df.to_csv(buffer)
		return buffer.getvalue()

def sendRichEmail(sender = None, receiver = None, password = None, subject = 'Default sendout', content = None, inline_images= None, attachment = None):
	"""
	:param sender: str email address from whom email is send
	:param receiver: str of comma seperated email addresses
	:param password: str
	:param subject: str subject of email
	:param content: str HTML formatted string (reference the images as <img src="cid:image1"> in the content)
	:param inline_images: list with paths to the images
	:param attachment: with path to the attachment
	:return: The email will be send
	"""
	# Check types and makes correct one
	# recipients
	if isinstance(receiver, list):
		receiverstr = ', '.join(map(str, receiver))
	# Create the root message and fill in the from, to, and subject headers
	msgRoot = MIMEMultipart('related')
	msgRoot['Subject'] = subject
	msgRoot['From'] = sender
	msgRoot['To'] = receiverstr
	msgRoot.preamble = 'This is a multi-part message in MIME format.'

	# Encapsulate the plain and HTML versions of the message body in an
	# 'alternative' part, so message agents can decide which they want to display.
	msgAlternative = MIMEMultipart('alternative')
	msgRoot.attach(msgAlternative)

	msgText = MIMEText('This is the alternative plain text message.')
	msgAlternative.attach(msgText)

	if attachment is not None:
		# Add attachment
		attachment = MIMEApplication(df_to_bytes(attachment))
		attachment['Content-Disposition'] = 'attachment; filename="{}"'.format('enriched_data.csv')
		msgRoot.attach(attachment)

	# We reference the image in the IMG SRC attribute by the ID we give it below
	msgText = MIMEText(content, 'html')
	msgAlternative.attach(msgText)

	if not inline_images is None:
		nr_images = len(inline_images)
		for i, img in enumerate(inline_images):
			fp = open(img, 'rb')
			msgImage = MIMEImage(fp.read())
			fp.close()

			# Define the image's ID as referenced above
			msgImage.add_header('Content-ID', '<image{}>'.format(i+1))
			msgRoot.attach(msgImage)

	# Sending the email
	with smtplib.SMTP('smtp.office365.com', 587) as server:
		server.ehlo()
		server.starttls()
		server.login(sender, password)
		server.sendmail(sender, receiver, msgRoot.as_string())

	print('Email with subject {} \nHas been sent to {} recipients'.format(subject, receiverstr.count('@')))

def sendEmailSmtpSsl(html_content, sender, recipient, username, password, smtp_server, port):
	# Sending an email with the predictions
	import smtplib, ssl
	from email.mime.text import MIMEText
	from email.mime.multipart import MIMEMultipart

	# Email configurations and content
	msg = MIMEMultipart()
	msg['Subject'] = "Stock buy advise"
	msg['From'] = sender

	part1 = MIMEText(html_content, 'html')
	msg.attach(part1)

	# Sending the email

	# Create a secure SSL context
	context = ssl.create_default_context()

	with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
		server.login(msg['From'], password)
		server.sendmail(msg['From'], recipient, msg.as_string())

	print('Email with predictions send')


def sendEmailSmtpTls(html_content, sender, recipient, username, password, smtp_server, port):
	# Sending an email with the predictions
	import smtplib, ssl
	from email.mime.text import MIMEText
	from email.mime.multipart import MIMEMultipart

	# Email configurations and content
	msg = MIMEMultipart()
	msg['Subject'] = "Stock buy advise"
	msg['From'] = sender

	part1 = MIMEText(html_content, 'html')
	msg.attach(part1)

	# Sending the email
	with smtplib.SMTP(smtp_server, port) as server:
		server.ehlo()
		server.starttls()
		server.login(username, password)
		server.sendmail(msg['From'], recipient, msg.as_string())

	print('email sent')


if __name__ == '__main__':
	html_content = """
	<html>
	  <head></head>
	  <body>
	  	<h3> High probability </h3>
	    {0}
	    <h4> Configurations </h4>
	    <p>
	    Minimal threshold:  <br>
	    Maximum stock price:  <br>
	    Days to expiration between  and  <br>
	    Strike price at least  higher than stock price <br>
	    </p>
	    <hr>
	    <h3> High profitability </h3>
	    
	    <h4> Configurations </h4>
	    <p>
	    Minimal threshold:  <br>
	    Maximum stock price:  <br>
	    Days to expiration between  and  <br>
	    Strike price at least  higher than stock price <br>
	    </p>
	    <p>
	    Or check the streamlit dashboard with the predictions: <br>
	    <a href="https://kaspersgit-option-trading-daily-predict-st-022izs.streamlitapp.com/">predictions dashboard</a> <br>
	    Just keep in mind this email and the dashboard are using different models <br>
	    </p>
	  </body>
	</html>
	"""

	sender='******'
	recipient='******'
	username=sender
	password='******'
	smtp_server='******'
	port=587
	sendEmailSmtpSsl(html_content, sender, recipient, username, password, smtp_server, port)

	sendEmailSmtpTls(html_content, sender, recipient, username, password, smtp_server, port)
