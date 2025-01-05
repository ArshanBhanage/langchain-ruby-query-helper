require 'faker'

# seed.rb

# Define roles
roles = %w[receptionist doctor]

# Create 5 doctors
5.times do
  User.create!(
    first_name: Faker::Name.first_name,
    last_name: Faker::Name.last_name,    
    email: Faker::Internet.email,
    password: "111111",
    password_confirmation: "111111",
    role: 1 # doctor
  )
end

# Create 1 receptionist

User.create!(
  first_name: Faker::Name.first_name,
  last_name: Faker::Name.last_name,  
  email: Faker::Internet.email,
  password: "111111",
  password_confirmation: "111111",
  role: 0 # receptionist
)

# Create patients
current_time = Time.now
past_6_months = 6.months.ago

1000.times do
  created_at = Faker::Time.between(from: past_6_months, to: current_time)
  updated_at = Faker::Time.between(from: created_at, to: current_time)

  Patient.create!(
    first_name: Faker::Name.first_name,
    last_name: Faker::Name.last_name,
    dob: Faker::Date.between(from: 60.years.ago, to: 18.years.ago),
    gender: ['Male', 'Female'].sample,
    address: Faker::Address.full_address,
    phone: Faker::PhoneNumber.phone_number,
    email: Faker::Internet.email,
    created_at: created_at,
    updated_at: updated_at
  )
end

# Create appointments for doctors
users = User.where(role: 1)
patients = Patient.all

500.times do
  user = users.sample
  patient = patients.sample

  created_at = Faker::Time.between(from: past_6_months, to: current_time)
  updated_at = Faker::Time.between(from: created_at, to: current_time)

  Appointment.create!(
    user_id: user.id,
    patient_id: patient.id,
    appointment_date: Faker::Time.between(from: 6.months.ago, to: Time.now),
    reason: Faker::Lorem.sentence,
    created_at: created_at,
    updated_at: updated_at    
  )
end
