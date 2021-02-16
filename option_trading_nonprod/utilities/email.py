# Send an HTML email with an embedded image and a plain text message for
# email clients that don't want to display the HTML.

from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import io

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
		receiver = ', '.join(map(str, receiver))
	# Create the root message and fill in the from, to, and subject headers
	msgRoot = MIMEMultipart('related')
	msgRoot['Subject'] = subject
	msgRoot['From'] = sender
	msgRoot['To'] = receiver
	msgRoot.preamble = 'This is a multi-part message in MIME format.'

	# Encapsulate the plain and HTML versions of the message body in an
	# 'alternative' part, so message agents can decide which they want to display.
	msgAlternative = MIMEMultipart('alternative')
	msgRoot.attach(msgAlternative)

	msgText = MIMEText('This is the alternative plain text message.')
	msgAlternative.attach(msgText)

	if attachment != None:
		# Add attachment
		attachment = MIMEApplication(df_to_bytes(attachment))
		attachment['Content-Disposition'] = 'attachment; filename="{}"'.format('enriched_data.csv')
		msgRoot.attach(attachment)

	# We reference the image in the IMG SRC attribute by the ID we give it below
	msgText = MIMEText(content, 'html')
	msgAlternative.attach(msgText)

	if len(inline_images) > 0:
		nr_images = len(inline_images)
		for i, img in enumerate(inline_images):
			fp = open(img, 'rb')
			msgImage = MIMEImage(fp.read())
			fp.close()

			# Define the image's ID as referenced above
			msgImage.add_header('Content-ID', '<image{}>'.format(i+1))
			msgRoot.attach(msgImage)

	# Send the email (this example assumes SMTP authentication is required)
	import smtplib
	with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
		smtp.ehlo()
		smtp.login(sender, password)
		smtp.sendmail(sender, receiver, msgRoot.as_string())
	print('Email with subject {} \nHas been sent to {} recipients'.format(subject, receiver.count('@')))
